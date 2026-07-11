using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// IModelSession 的远程实现：经 RuntimeClient 把 Run 派发到 runtime（loopback / 子进程）。
//   以「模型路径」为稳定键交给 client——client 内部映射 path→sessionId 并在子进程 respawn 后按 path 重载，
//   故本会话跨 respawn 保持有效（sessionId 变化对上层透明）。
//   输入元数据（名→形状）在 Load 时一次性取回本地镜像（同模型跨 respawn 不变），HasInput/InputShape 走本地、不往返。
//   与 InProcessModelSession 对编排层完全等价——切换实现即切换「推理在哪跑」，编排一行不改。
internal sealed class RemoteModelSession : IModelSession
{
    readonly RuntimeClient mClient;
    readonly string mPath;
    readonly IReadOnlyDictionary<string, int[]> mInputs;

    RemoteModelSession(RuntimeClient client, string path, IReadOnlyDictionary<string, int[]> inputs)
    {
        mClient = client;
        mPath = path;
        mInputs = inputs;
    }

    public static RemoteModelSession Load(RuntimeClient client, string modelPath)
    {
        var inputs = client.Load(modelPath);
        return new RemoteModelSession(client, modelPath, inputs);
    }

    public bool HasInput(string name) => mInputs.ContainsKey(name);

    public IReadOnlyList<int> InputShape(string name) => mInputs[name];

    public IReadOnlyList<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        => mClient.Run(mPath, inputs);

    // 释放 host 侧会话（子进程内），避免单 runtime 生命周期内的会话泄漏；client 生命周期本身由缓存/引擎管理。
    public void Dispose() => mClient.Release(mPath);
}
