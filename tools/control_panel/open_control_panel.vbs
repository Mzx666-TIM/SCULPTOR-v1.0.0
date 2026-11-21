Set fso = CreateObject("Scripting.FileSystemObject")
Set ws = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
root = fso.GetParentFolderName(fso.GetParentFolderName(scriptDir))
pyw = root & "\.venv\Scripts\pythonw.exe"
ws.Run """" & pyw & """ """ & scriptDir & "\desktop_panel.pyw""", 0, False
