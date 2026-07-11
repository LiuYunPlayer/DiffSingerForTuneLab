using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Hashing;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// DiffSinger 张量缓存：把一个 ONNX 模型调用的输出按「模型文件哈希 + 序列化输入」为键缓存到磁盘，
//   反复合成（撤销重做、重开工程、改动不影响某块、跨块/跨说话人共享 linguistic 等）时直接复用、免重算。
// 忠实移植 OpenUtau DiffSingerCache（序列化/反序列化、文件格式与按 name 排序求键一致），差异仅在：
//   · 哈希用 System.IO.Hashing.XxHash64（OpenUtau 用 K4os.Hash.xxHash），数值不同但本插件不需跨工具兼容；
//   · 缓存目录取插件独立用户数据根 UserDataRoot/Cache（OpenUtau 用 PathManager.Inst.CachePath）。
// 另附编排封装：Run（建键→Load→未命中则模型 Run + Save，返回脱离原生内存的托管张量）、
//   Clone（把模型原生输出深拷成托管，供未命中/禁用时安全返回）、HashFile（模型 identifier）、EnforceSizeLimit（LRU 逐出）。
public sealed class DiffSingerTensorCache
{
    const string FormatHeader = "TENSORCACHE";

    readonly ulong mHash;
    readonly string mFilename;

    public ulong Hash => mHash;
    public string Filename => mFilename;

    // 缓存目录：插件用户数据根下的 Cache（与 Voices/Vocoders 并列）。
    public static string CacheDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Cache");

    public DiffSingerTensorCache(ulong identifier, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(identifier);
            foreach (var onnxValue in inputs.OrderBy(v => v.Name, StringComparer.InvariantCulture))
                SerializeNamedOnnxValue(writer, onnxValue);
        }

        mHash = XxHash64.HashToUInt64(stream.ToArray());
        mFilename = $"ds-{mHash:x16}.tensorcache";
    }

    // —— 编排封装 ——
    // 推理串行化 + 会话退役机制已内聚进 InProcessModelSession（设备级约束随「推理在哪跑」走）；
    //   本类只管缓存编排与张量序列化，不再持有原生会话或全局锁。

    // 跑一个模型并经缓存：命中直接返回缓存输出；未命中则 Run + Save。model.Run 已返回脱离原生内存的托管张量。
    //   enabled=false 时跳过磁盘缓存，直接返回 model.Run 的托管输出。
    public static IReadOnlyList<NamedOnnxValue> Run(
        IModelSession model, ulong identifier, IReadOnlyCollection<NamedOnnxValue> inputs, bool enabled)
    {
        if (!enabled)
            return model.Run(inputs);

        var cache = new DiffSingerTensorCache(identifier, inputs);
        var loaded = cache.Load();
        if (loaded != null)
            return loaded;

        var result = model.Run(inputs);
        cache.Save(result);
        return result;
    }

    // 把（可能由原生 OrtValue 支撑的）输出深拷为托管 DenseTensor，使其在原生集合 Dispose 后仍可安全读取。
    //   复用序列化/反序列化的类型分支（往返一次内存流），零重复代码；相对扩散推理开销可忽略。
    public static List<NamedOnnxValue> Clone(IEnumerable<NamedOnnxValue> values)
    {
        var list = new List<NamedOnnxValue>();
        foreach (var v in values)
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
                SerializeNamedOnnxValue(w, v);
            ms.Position = 0;
            using var r = new BinaryReader(ms);
            list.Add(DeserializeNamedOnnxValue(r));
        }
        return list;
    }

    // 模型 identifier：.onnx 文件内容的 XxHash64（流式、不整体载入内存）。加载时算一次缓存进字段，
    //   用作缓存键的一部分，区分不同模型（同输入不同权重不撞键），且模型文件更换即自动失效。
    public static ulong HashFile(string path)
    {
        var h = new XxHash64();
        using var fs = File.OpenRead(path);
        h.Append(fs);
        return h.GetCurrentHashAsUInt64();
    }

    // LRU 体积上限逐出：缓存目录超过上限时，按最近访问时间删最旧的 .tensorcache 直到回落。maxSizeMb<=0 视作不限制。
    //   尽力而为，任何 IO 异常吞掉（逐出失败不应影响合成）。
    public static void EnforceSizeLimit(long maxSizeMb)
    {
        if (maxSizeMb <= 0)
            return;
        try
        {
            var dir = CacheDirectory;
            if (!Directory.Exists(dir))
                return;
            var files = new DirectoryInfo(dir).GetFiles("*.tensorcache");
            long total = files.Sum(f => f.Length);
            long limit = maxSizeMb * 1024L * 1024L;
            if (total <= limit)
                return;
            foreach (var f in files.OrderBy(f => f.LastAccessTimeUtc))
            {
                try { long len = f.Length; f.Delete(); total -= len; } catch { }
                if (total <= limit)
                    break;
            }
        }
        catch { }
    }

    public IReadOnlyList<NamedOnnxValue>? Load()
    {
        var cachePath = Path.Join(CacheDirectory, mFilename);
        if (!File.Exists(cachePath))
            return null;

        var result = new List<NamedOnnxValue>();
        try
        {
            using (var stream = new FileStream(cachePath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                if (reader.ReadString() != FormatHeader)
                    throw new InvalidDataException($"[TensorCache] 缓存文件头异常：{mFilename}。");
                result = ReadValues(reader);
            }
        }
        catch (Exception e)
        {
            TuneLabContext.Global.GetLogger().Warning($"DiffSinger：反序列化缓存 {mFilename} 失败、丢弃重算：{e.Message}");
            Delete();
            return null;
        }

        // 命中即「访问」：显式刷新访问时间，令 LRU 逐出以真实使用近度排序（不依赖 NTFS 自动 last-access 策略）。
        try { File.SetLastAccessTimeUtc(cachePath, DateTime.UtcNow); } catch { }
        return result;
    }

    public void Delete()
    {
        var cachePath = Path.Join(CacheDirectory, mFilename);
        if (File.Exists(cachePath))
        {
            try { File.Delete(cachePath); } catch { }
        }
    }

    public void Save(IReadOnlyCollection<NamedOnnxValue> outputs)
    {
        Directory.CreateDirectory(CacheDirectory);
        var cachePath = Path.Join(CacheDirectory, mFilename);
        using var stream = new FileStream(cachePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        writer.Write(FormatHeader);
        WriteValues(writer, outputs);
    }

    // —— 张量组序列化（单一来源）：磁盘缓存与 IPC 编解码共用；格式 = [count][value...]（不含缓存文件头）。 ——
    internal static void WriteValues(BinaryWriter writer, IReadOnlyCollection<NamedOnnxValue> values)
    {
        writer.Write(values.Count);
        foreach (var v in values)
            SerializeNamedOnnxValue(writer, v);
    }

    internal static List<NamedOnnxValue> ReadValues(BinaryReader reader)
    {
        var count = reader.ReadInt32();
        var list = new List<NamedOnnxValue>(count);
        for (var i = 0; i < count; ++i)
            list.Add(DeserializeNamedOnnxValue(reader));
        return list;
    }

    static void SerializeNamedOnnxValue(BinaryWriter writer, NamedOnnxValue namedOnnxValue)
    {
        if (namedOnnxValue.ValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            throw new NotSupportedException(
                $"[TensorCache] 仅支持张量类型 {OnnxValueType.ONNX_TYPE_TENSOR}，遇 {namedOnnxValue.ValueType}。");
        writer.Write(namedOnnxValue.Name);
        var tensorBase = (TensorBase)namedOnnxValue.Value;
        var elementType = tensorBase.GetTypeInfo().ElementType;
        writer.Write((int)elementType);
        switch (elementType)
        {
            case TensorElementType.Float: SerializeTensor(writer, namedOnnxValue.AsTensor<float>()); break;
            case TensorElementType.UInt8: SerializeTensor(writer, namedOnnxValue.AsTensor<byte>()); break;
            case TensorElementType.Int8: SerializeTensor(writer, namedOnnxValue.AsTensor<sbyte>()); break;
            case TensorElementType.UInt16: SerializeTensor(writer, namedOnnxValue.AsTensor<ushort>()); break;
            case TensorElementType.Int16: SerializeTensor(writer, namedOnnxValue.AsTensor<short>()); break;
            case TensorElementType.Int32: SerializeTensor(writer, namedOnnxValue.AsTensor<int>()); break;
            case TensorElementType.Int64: SerializeTensor(writer, namedOnnxValue.AsTensor<long>()); break;
            case TensorElementType.String: SerializeTensor(writer, namedOnnxValue.AsTensor<string>()); break;
            case TensorElementType.Bool: SerializeTensor(writer, namedOnnxValue.AsTensor<bool>()); break;
            case TensorElementType.Float16: SerializeTensor(writer, namedOnnxValue.AsTensor<Float16>()); break;
            case TensorElementType.Double: SerializeTensor(writer, namedOnnxValue.AsTensor<double>()); break;
            case TensorElementType.UInt32: SerializeTensor(writer, namedOnnxValue.AsTensor<uint>()); break;
            case TensorElementType.UInt64: SerializeTensor(writer, namedOnnxValue.AsTensor<ulong>()); break;
            case TensorElementType.BFloat16: SerializeTensor(writer, namedOnnxValue.AsTensor<BFloat16>()); break;
            default:
                throw new NotSupportedException($"[TensorCache] 不支持的张量元素类型：{elementType}。");
        }
    }

    static void SerializeTensor<T>(BinaryWriter writer, Tensor<T> tensor)
    {
        if (tensor.IsReversedStride)
            throw new NotSupportedException("[TensorCache] 不支持反序步幅张量。");
        writer.Write(tensor.Rank);
        foreach (var dim in tensor.Dimensions)
            writer.Write(dim);
        var size = (int)tensor.Length;
        writer.Write(size);
        if (typeof(T) == typeof(string))
        {
            foreach (var element in tensor.ToArray())
                writer.Write(element?.ToString() ?? string.Empty);
        }
        else
        {
            var data = new byte[size * tensor.GetTypeInfo().TypeSize];
            Buffer.BlockCopy(tensor.ToArray(), 0, data, 0, data.Length);
            writer.Write(data);
        }
    }

    static NamedOnnxValue DeserializeNamedOnnxValue(BinaryReader reader)
    {
        var name = reader.ReadString();
        var dtype = (TensorElementType)reader.ReadInt32();
        var rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (var i = 0; i < rank; ++i)
            shape[i] = reader.ReadInt32();
        var size = reader.ReadInt32();
        switch (dtype)
        {
            case TensorElementType.Float: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<float>(reader, size, sizeof(float), shape));
            case TensorElementType.UInt8: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<byte>(reader, size, sizeof(byte), shape));
            case TensorElementType.Int8: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<sbyte>(reader, size, sizeof(sbyte), shape));
            case TensorElementType.UInt16: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<ushort>(reader, size, sizeof(ushort), shape));
            case TensorElementType.Int16: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<short>(reader, size, sizeof(short), shape));
            case TensorElementType.Int32: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<int>(reader, size, sizeof(int), shape));
            case TensorElementType.Int64: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<long>(reader, size, sizeof(long), shape));
            case TensorElementType.String:
            {
                Tensor<string> tensor = new DenseTensor<string>(size);
                for (var i = 0; i < size; ++i)
                    tensor[i] = reader.ReadString();
                tensor = tensor.Reshape(shape);
                return NamedOnnxValue.CreateFromTensor(name, tensor);
            }
            case TensorElementType.Bool: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<bool>(reader, size, sizeof(bool), shape));
            case TensorElementType.Float16: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<Float16>(reader, size, sizeof(ushort), shape));
            case TensorElementType.Double: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<double>(reader, size, sizeof(double), shape));
            case TensorElementType.UInt32: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<uint>(reader, size, sizeof(uint), shape));
            case TensorElementType.UInt64: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<ulong>(reader, size, sizeof(ulong), shape));
            case TensorElementType.BFloat16: return NamedOnnxValue.CreateFromTensor(name, DeserializeTensor<BFloat16>(reader, size, sizeof(ushort), shape));
            default:
                throw new NotSupportedException($"[TensorCache] 不支持的张量元素类型：{dtype}。");
        }
    }

    static Tensor<T> DeserializeTensor<T>(BinaryReader reader, int size, int typeSize, ReadOnlySpan<int> shape)
    {
        var bytes = reader.ReadBytes(size * typeSize);
        var data = new T[size];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return new DenseTensor<T>(data, shape);
    }
}
