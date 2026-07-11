using System;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace DiffSingerForTuneLab;

// Win32 Job Object（JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE）：把 MLRuntime 子进程塞进 Job，
//   Job 句柄随本进程（宿主 TuneLab）消亡而关闭 → OS 强制杀 Job 内所有子进程。
//   哪怕宿主硬崩（不走任何清理），OS 也兜底杀掉子进程，不留孤儿。这是治孤儿的主保险（管道 EOF 自杀为副）。
[SupportedOSPlatform("windows")]
internal sealed class JobObject : IDisposable
{
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string? lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr hJob, int infoClass,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInfo, int cbJobObjectInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr hObject);

    const int JobObjectExtendedLimitInformation = 9;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS
    {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    IntPtr mHandle;

    public JobObject()
    {
        mHandle = CreateJobObject(IntPtr.Zero, null);
        if (mHandle == IntPtr.Zero)
            throw new InvalidOperationException($"CreateJobObject 失败（Win32 {Marshal.GetLastWin32Error()}）");

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(mHandle, JobObjectExtendedLimitInformation, ref info, Marshal.SizeOf(info)))
            throw new InvalidOperationException($"SetInformationJobObject 失败（Win32 {Marshal.GetLastWin32Error()}）");
    }

    public void AssignProcess(IntPtr processHandle)
    {
        if (!AssignProcessToJobObject(mHandle, processHandle))
            throw new InvalidOperationException($"AssignProcessToJobObject 失败（Win32 {Marshal.GetLastWin32Error()}）");
    }

    public void Dispose()
    {
        if (mHandle != IntPtr.Zero)
        {
            CloseHandle(mHandle);   // 关 Job 句柄 → KILL_ON_JOB_CLOSE 触发、Job 内子进程被杀
            mHandle = IntPtr.Zero;
        }
    }
}
