<#
Native folder picker for the control panel.

Uses the Windows Common Item Dialog (IFileOpenDialog with FOS_PICKFOLDERS) -
the same dialog Explorer shows - instead of the legacy folder browser that
Windows Forms exposes: the legacy one expands every folder inline for machines
whose shell is configured that way, which is not what a modern control panel
should look like.

Prints the selected path on stdout, or an empty line when the user cancels.
Exits non-zero when the dialog cannot be created at all.
#>
param(
    [string]$Initial = "",
    [string]$Title = "Select output folder",
    [switch]$SelfTest
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace SculptorNative
{
    [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IFileDialog
    {
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
        void SetFileTypeIndex(uint iFileType);
        void GetFileTypeIndex(out uint piFileType);
        void Advise(IntPtr pfde, out uint pdwCookie);
        void Unadvise(uint dwCookie);
        void SetOptions(uint fos);
        void GetOptions(out uint pfos);
        void SetDefaultFolder(IShellItem psi);
        void SetFolder(IShellItem psi);
        void GetFolder(out IShellItem ppsi);
        void GetCurrentSelection(out IShellItem ppsi);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
        void GetResult(out IShellItem ppsi);
        void AddPlace(IShellItem psi, int fdap);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
        void Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr pFilter);
    }

    [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IShellItem
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem ppsi);
        void GetDisplayName(uint sigdnName,
                            [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
        void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
        void Compare(IShellItem psi, uint hint, out int piOrder);
    }

    [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    public class FileOpenDialog
    {
    }

    public static class FolderPicker
    {
        private const uint FOS_PICKFOLDERS = 0x00000020;
        private const uint FOS_FORCEFILESYSTEM = 0x00000040;
        private const uint FOS_PATHMUSTEXIST = 0x00000800;
        private const uint SIGDN_FILESYSPATH = 0x80058000;
        private const int ERROR_CANCELLED = unchecked((int)0x800704C7);
        private static readonly Guid IID_IShellItem =
            new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void SHCreateItemFromParsingName(
            string pszPath, IntPtr pbc, ref Guid riid,
            [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

        private static IFileDialog CreateDialog(string title)
        {
            IFileDialog dialog = (IFileDialog)new FileOpenDialog();
            uint options;
            dialog.GetOptions(out options);
            dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM
                              | FOS_PATHMUSTEXIST);
            if (!string.IsNullOrEmpty(title))
            {
                dialog.SetTitle(title);
            }
            return dialog;
        }

        public static string Pick(string initialFolder, string title)
        {
            IFileDialog dialog = CreateDialog(title);
            if (!string.IsNullOrEmpty(initialFolder) && Directory.Exists(initialFolder))
            {
                Guid iid = IID_IShellItem;
                IShellItem folder;
                SHCreateItemFromParsingName(initialFolder, IntPtr.Zero, ref iid,
                                            out folder);
                dialog.SetFolder(folder);
            }
            int hr = dialog.Show(IntPtr.Zero);
            if (hr == ERROR_CANCELLED)
            {
                return "";
            }
            Marshal.ThrowExceptionForHR(hr);
            IShellItem result;
            dialog.GetResult(out result);
            string path;
            result.GetDisplayName(SIGDN_FILESYSPATH, out path);
            return path ?? "";
        }

        /// <summary>Creates the dialog and reads its options back, without showing it.</summary>
        public static string SelfTest()
        {
            IFileDialog dialog = CreateDialog("SCULPTOR self test");
            uint options;
            dialog.GetOptions(out options);
            bool folders = (options & FOS_PICKFOLDERS) != 0;
            bool filesystem = (options & FOS_FORCEFILESYSTEM) != 0;
            return folders && filesystem ? "ok" : "options-not-applied";
        }
    }
}
'@

if ($SelfTest) {
    Write-Output ([SculptorNative.FolderPicker]::SelfTest())
    exit 0
}

$selected = [SculptorNative.FolderPicker]::Pick($Initial, $Title)
Write-Output $selected
exit 0
