using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// IModelSession 的远程实现：经 RuntimeClient 把 Run 派发到 runtime（loopback / 子进程）。
//   元数据（输入口名→形状）在 LoadModel 时一次性取回本地镜像，HasInput/InputShape 走本地、不往返。
//   与 InProcessModelSession 对编排层完全等价——切换实现即切换「推理在哪跑」，编排一行不改。
internal sealed class RemoteModelSession : IModelSession
{
    readonly RuntimeClient mClient;
    readonly int mSessionId;
    readonly IReadOnlyDictionary<string, int[]> mInputs;

    RemoteModelSession(RuntimeClient client, int sessionId, IReadOnlyDictionary<string, int[]> inputs)
    {
        mClient = client;
        mSessionId = sessionId;
        mInputs = inputs;
    }

    public static RemoteModelSession Load(RuntimeClient client, string modelPath)
    {
        var (id, inputs) = client.LoadModel(modelPath);
        return new RemoteModelSession(client, id, inputs);
    }

    public bool HasInput(string name) => mInputs.ContainsKey(name);

    public IReadOnlyList<int> InputShape(string name) => mInputs[name];

    public IReadOnlyList<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        => mClient.Run(mSessionId, inputs);

    // 远程会话不拥有本进程原生资源（在 runtime 侧）；runtime 生命周期由缓存/引擎统一管理（P4）。
    //   TODO(P4)：换 runtime / 关闭时经 client 发 Release 释放 host 侧会话，避免单 runtime 生命周期内的会话泄漏。
    public void Dispose() { }
}
