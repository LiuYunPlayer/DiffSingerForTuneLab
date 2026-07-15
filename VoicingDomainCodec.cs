using System;

namespace DiffSingerForTuneLab;

// mulaw voicing 域编解码（模型边界"三明治"转换）：线上值 ↔ dB 语义值。
// 背景：mulaw 声库（fork feat/mulaw-voicing）binarize 期把谐波 RMS 压 mu-law（μ 缺省 255）后线性映射到
//   [-96, 0]——与 db 声库同数值域（上游 API 冻结下的伪装），但语义是压缩振幅而非分贝。
// 插件契约：Delta 公式 / 回显 / clamp 恒在 dB 语义层（十二次幂三锚定是感知规格、与模型编码无关，
//   见 schema §14.2），仅在模型边界做双向转换——两种表示都是振幅的单调双射，转换精确无损。
//   db 声库（含无 voicing_domain 字段的历史声库）codec = null，走原路径、逐比特不变。
sealed class VoicingDomainCodec
{
    const double Floor = -96.0;   // 数字静音底（两域共用的线上下界；mulaw 域上 −96 ↔ 振幅精确 0）

    readonly double mMu;
    readonly double mLog1pMu;     // ln(1+μ)

    internal VoicingDomainCodec(double mu) { mMu = mu; mLog1pMu = Math.Log(1 + mu); }

    // 仅 mulaw 需要转换；db 返回 null ⇒ 调用方走原路径。裸参签名（不吃 VoicebankConfig）：
    //   本文件保持零依赖，可被 SmokeTest 等零 TuneLab 依赖的工具链接（csproj Compile Include）。
    public static VoicingDomainCodec? For(string domain, double mu)
        => domain == "mulaw" ? new VoicingDomainCodec(mu) : null;

    // 线上值 → dB：wire ∈ [-96,0] → mulaw 归一 m ∈ [0,1] → 振幅 a = (e^{m·ln(1+μ)} − 1)/μ → 20·log10(a)。
    //   a=0 时 log10 → −∞，由 Max 落回地板 −96。
    public float WireToDb(float wire)
    {
        double m = Math.Clamp((wire - Floor) / -Floor, 0, 1);
        double a = (Math.Exp(m * mLog1pMu) - 1) / mMu;
        return (float)Math.Max(20 * Math.Log10(a), Floor);
    }

    // dB → 线上值：a = 10^{dB/20} → wire = −Floor·ln(1+μa)/ln(1+μ) + Floor。
    //   触底特判：dB ≤ −96 → 线上精确 −96（mulaw 域对应振幅精确 0 = 真·静音，而非 clamp 余响 ≈−95.93）。
    public float DbToWire(float db)
    {
        if (db <= Floor) return (float)Floor;
        double a = Math.Pow(10, Math.Min(db, 0) / 20.0);
        return (float)(-Floor * Math.Log(1 + mMu * a) / mLog1pMu + Floor);
    }
}
