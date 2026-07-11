using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// 一个已加载 ONNX 模型的推理会话抽象——把编排层与「推理在哪跑」解耦。
//   P1 只有进程内实现（InProcessModelSession，直跑原生 InferenceSession）；
//   P4 将追加走 IPC 的远程实现（RemoteModelSession），编排层一行不改。
// 刻意不暴露 onnxruntime 的 NodeMetadata/InputMetadata：编排层只需要「有没有某输入口」+「某输入口的形状」，
//   用 HasInput/InputShape 两个原语覆盖，远程侧不必重建 NodeMetadata（它跨进程拿不到），接口一次成型。
// Run 契约：返回已脱离原生内存的托管张量——设备级串行、退役检查、原生输出深拷都是实现内部的事，编排层拿到即可安全延后读取。
public interface IModelSession : IDisposable
{
    // 是否声明了名为 name 的输入口（stock/fork 模型输入集不同，编排按此条件构造张量）。
    bool HasInput(string name);
    // 输入口 name 的形状（轴长，动态轴为 -1）；name 不存在则抛。
    IReadOnlyList<int> InputShape(string name);
    // 跑一次推理，返回脱离原生内存的托管输出（可在调用处延后读取）。
    IReadOnlyList<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs);
}

// 进程内实现：直跑本进程加载的原生 InferenceSession。
//   串行化 + 退役机制（原在 DiffSingerTensorCache）随会话内聚到此：
//   · 设备级串行锁 sRunLock：DirectML 下 Run 不可并发——不仅同会话，同 GPU 上不同会话并发 Run 亦崩，
//     故所有实例共享同一把静态锁跨会话全局串行；CPU EP 亦中性偏好（单次 Run 已用满 intra-op 线程池）。
//   · 退役标记 mRetired：会话随缓存释放前先在锁内置位，令其后任何 Run 干净抛 ObjectDisposedException，
//     而非把已释放的原生句柄喂进 Run（→ AccessViolationException）。Dispose 全程持锁 ⇒ 释放不与在飞 Run 并发。
internal sealed class InProcessModelSession : IModelSession
{
    static readonly object sRunLock = new();

    readonly InferenceSession mSession;
    bool mRetired;

    public InProcessModelSession(InferenceSession session) => mSession = session;

    public bool HasInput(string name) => mSession.InputMetadata.ContainsKey(name);

    public IReadOnlyList<int> InputShape(string name) => mSession.InputMetadata[name].Dimensions;

    public IReadOnlyList<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        lock (sRunLock)
        {
            if (mRetired)
                throw new ObjectDisposedException(nameof(InferenceSession), "DiffSinger：模型会话已随缓存释放，推理取消。");
            using var raw = mSession.Run(inputs);
            return TensorCodec.Clone(raw);   // 先脱离原生内存再返回
        }
    }

    // 在设备级锁内退役并释放：持锁即保证当前无 Run 在飞（Run 全程持同一把锁），释放不会与在飞推理并发触发 use-after-free。
    public void Dispose()
    {
        lock (sRunLock)
        {
            mRetired = true;
            mSession.Dispose();
        }
    }
}
