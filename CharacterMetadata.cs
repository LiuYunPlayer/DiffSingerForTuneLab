using System;
using System.Collections.Generic;
using System.IO;
using YamlDotNet.Serialization;

namespace DiffSingerForTuneLab;

// 读取声库的展示元数据：优先结构化的 character.yaml，回退 character.txt（key=value 行，
// 经典 UTAU 形态；无 key 的首个非空行视作名字）。两者皆缺或解析失败则各字段为 null。
// 图片字段返回相对声库根的文件名，由调用方拼绝对路径并校验存在。
internal static class CharacterMetadata
{
    public sealed record Result(string? Name, string? Author, string? ImageFile);

    public static Result Read(string bankDir)
    {
        var yaml = Path.Combine(bankDir, "character.yaml");
        if (File.Exists(yaml))
            return FromYaml(yaml);

        var txt = Path.Combine(bankDir, "character.txt");
        if (File.Exists(txt))
            return FromTxt(txt);

        return new Result(null, null, null);
    }

    static Result FromYaml(string path)
    {
        try
        {
            var deserializer = new DeserializerBuilder().Build();
            var map = deserializer.Deserialize<Dictionary<string, object?>>(File.ReadAllText(path));
            if (map is null)
                return new Result(null, null, null);

            // 标量值反序列化为 string；嵌套结构（如 subbanks）为 Dictionary/List，此处只取标量字段。
            var fields = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in map)
                if (kvp.Value is string s)
                    fields[kvp.Key] = s;

            fields.TryGetValue("name", out var name);
            if (!fields.TryGetValue("author", out var author))
                fields.TryGetValue("voice", out author);
            if (!fields.TryGetValue("image", out var image))
                fields.TryGetValue("portrait", out image);

            return new Result(name, author, image);
        }
        catch
        {
            return new Result(null, null, null);
        }
    }

    static Result FromTxt(string path)
    {
        string? name = null, author = null, image = null;
        try
        {
            foreach (var raw in File.ReadAllLines(path))
            {
                var line = raw.Trim();
                if (line.Length == 0)
                    continue;

                int eq = line.IndexOf('=');
                if (eq > 0)
                {
                    var key = line[..eq].Trim().ToLowerInvariant();
                    var value = line[(eq + 1)..].Trim();
                    switch (key)
                    {
                        case "name": name ??= value; break;
                        case "author":
                        case "voice": author ??= value; break;
                        case "image":
                        case "portrait": image ??= value; break;
                    }
                }
                else
                {
                    name ??= line;   // 无 key 的首个非空行 = 名字
                }
            }
        }
        catch
        {
            // 读取失败按各字段缺省（null）处理。
        }
        return new Result(name, author, image);
    }
}
