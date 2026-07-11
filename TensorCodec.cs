using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DiffSingerForTuneLab;

// ONNX 张量（NamedOnnxValue）二进制编解码——纯函数、零 TuneLab 依赖，故可经链接文件同时被插件与 MLRuntime.exe 复用。
//   格式忠实沿用 OpenUtau DiffSingerCache 的张量序列化（磁盘缓存与 IPC 线协议单一来源）；已被磁盘缓存长期往返证明无损。
//   从 DiffSingerTensorCache 抽出：缓存类保留磁盘/键/LRU 等编排（依赖 TuneLab），本类只留可移植的张量比特搬运。
internal static class TensorCodec
{
    // —— 单值（无 count 前缀）：供缓存键按序写入、Clone 往返。 ——
    public static void WriteValue(BinaryWriter writer, NamedOnnxValue namedOnnxValue)
    {
        if (namedOnnxValue.ValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            throw new NotSupportedException(
                $"[TensorCodec] 仅支持张量类型 {OnnxValueType.ONNX_TYPE_TENSOR}，遇 {namedOnnxValue.ValueType}。");
        writer.Write(namedOnnxValue.Name);
        var tensorBase = (TensorBase)namedOnnxValue.Value;
        var elementType = tensorBase.GetTypeInfo().ElementType;
        writer.Write((int)elementType);
        switch (elementType)
        {
            case TensorElementType.Float: WriteTensor(writer, namedOnnxValue.AsTensor<float>()); break;
            case TensorElementType.UInt8: WriteTensor(writer, namedOnnxValue.AsTensor<byte>()); break;
            case TensorElementType.Int8: WriteTensor(writer, namedOnnxValue.AsTensor<sbyte>()); break;
            case TensorElementType.UInt16: WriteTensor(writer, namedOnnxValue.AsTensor<ushort>()); break;
            case TensorElementType.Int16: WriteTensor(writer, namedOnnxValue.AsTensor<short>()); break;
            case TensorElementType.Int32: WriteTensor(writer, namedOnnxValue.AsTensor<int>()); break;
            case TensorElementType.Int64: WriteTensor(writer, namedOnnxValue.AsTensor<long>()); break;
            case TensorElementType.String: WriteTensor(writer, namedOnnxValue.AsTensor<string>()); break;
            case TensorElementType.Bool: WriteTensor(writer, namedOnnxValue.AsTensor<bool>()); break;
            case TensorElementType.Float16: WriteTensor(writer, namedOnnxValue.AsTensor<Float16>()); break;
            case TensorElementType.Double: WriteTensor(writer, namedOnnxValue.AsTensor<double>()); break;
            case TensorElementType.UInt32: WriteTensor(writer, namedOnnxValue.AsTensor<uint>()); break;
            case TensorElementType.UInt64: WriteTensor(writer, namedOnnxValue.AsTensor<ulong>()); break;
            case TensorElementType.BFloat16: WriteTensor(writer, namedOnnxValue.AsTensor<BFloat16>()); break;
            default:
                throw new NotSupportedException($"[TensorCodec] 不支持的张量元素类型：{elementType}。");
        }
    }

    public static NamedOnnxValue ReadValue(BinaryReader reader)
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
            case TensorElementType.Float: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<float>(reader, size, sizeof(float), shape));
            case TensorElementType.UInt8: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<byte>(reader, size, sizeof(byte), shape));
            case TensorElementType.Int8: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<sbyte>(reader, size, sizeof(sbyte), shape));
            case TensorElementType.UInt16: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<ushort>(reader, size, sizeof(ushort), shape));
            case TensorElementType.Int16: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<short>(reader, size, sizeof(short), shape));
            case TensorElementType.Int32: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<int>(reader, size, sizeof(int), shape));
            case TensorElementType.Int64: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<long>(reader, size, sizeof(long), shape));
            case TensorElementType.String:
            {
                Tensor<string> tensor = new DenseTensor<string>(size);
                for (var i = 0; i < size; ++i)
                    tensor[i] = reader.ReadString();
                tensor = tensor.Reshape(shape);
                return NamedOnnxValue.CreateFromTensor(name, tensor);
            }
            case TensorElementType.Bool: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<bool>(reader, size, sizeof(bool), shape));
            case TensorElementType.Float16: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<Float16>(reader, size, sizeof(ushort), shape));
            case TensorElementType.Double: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<double>(reader, size, sizeof(double), shape));
            case TensorElementType.UInt32: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<uint>(reader, size, sizeof(uint), shape));
            case TensorElementType.UInt64: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<ulong>(reader, size, sizeof(ulong), shape));
            case TensorElementType.BFloat16: return NamedOnnxValue.CreateFromTensor(name, ReadTensor<BFloat16>(reader, size, sizeof(ushort), shape));
            default:
                throw new NotSupportedException($"[TensorCodec] 不支持的张量元素类型：{dtype}。");
        }
    }

    // —— 张量组（count + value...）：磁盘缓存与 IPC 复用；不含缓存文件头。 ——
    public static void WriteValues(BinaryWriter writer, IReadOnlyCollection<NamedOnnxValue> values)
    {
        writer.Write(values.Count);
        foreach (var v in values)
            WriteValue(writer, v);
    }

    public static List<NamedOnnxValue> ReadValues(BinaryReader reader)
    {
        var count = reader.ReadInt32();
        var list = new List<NamedOnnxValue>(count);
        for (var i = 0; i < count; ++i)
            list.Add(ReadValue(reader));
        return list;
    }

    // 把（可能由原生 OrtValue 支撑的）输出深拷为托管 DenseTensor，使其在原生集合 Dispose 后仍可安全读取（往返一次内存流）。
    public static List<NamedOnnxValue> Clone(IEnumerable<NamedOnnxValue> values)
    {
        var list = new List<NamedOnnxValue>();
        foreach (var v in values)
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
                WriteValue(w, v);
            ms.Position = 0;
            using var r = new BinaryReader(ms);
            list.Add(ReadValue(r));
        }
        return list;
    }

    static void WriteTensor<T>(BinaryWriter writer, Tensor<T> tensor)
    {
        if (tensor.IsReversedStride)
            throw new NotSupportedException("[TensorCodec] 不支持反序步幅张量。");
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

    static Tensor<T> ReadTensor<T>(BinaryReader reader, int size, int typeSize, ReadOnlySpan<int> shape)
    {
        var bytes = reader.ReadBytes(size * typeSize);
        var data = new T[size];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return new DenseTensor<T>(data, shape);
    }
}
