; VideoHighlighter Windows setup (Inno Setup 6.1+).
;
; Built in CI by build-release.yaml, in the same job that produces the payload,
; because only that job knows the volume count and the sizes:
;   ISCC.exe /DAppVersion= /DTag= /DVolumes= /DArchiveMB= /DInstalledMB= \
;            packaging\installer\videohighlighter.iss
;
; TWO PAYLOAD SOURCES, one install-time path. Without /DEmbeddedArchive, Setup
; downloads .7z volumes from the release it was built for; with it, that archive
; travels inside Setup.exe and nothing is downloaded. Everything after the bytes
; land -- unpacking, shortcuts, uninstall -- is the same code either way.
;
; Downloading is not a preference, it is the 2 GB cap on a GitHub release asset:
; the Windows build is ~2.7 GB compressed, so a public release has to split it,
; and an installer that carried it whole could not be attached. What the
; downloader removes is the part people actually got wrong -- the second volume.
; Before this, the release offered .7z.001 and .7z.002 and a bootstrap zip whose
; instructions asked you to extract it and double-click a .bat; forgetting the
; second part was the most common install failure. Here the volume list is
; compiled in, so there is nothing to forget and nothing to extract by hand.
; Where the file is delivered by other means and no such cap applies, the same
; installer carries it instead and installs offline.
;
; PER-USER, NO ADMIN: the app keeps its cache, debug.log and config.yaml beside
; its own executable whenever that folder is writable (modules/system/app_paths.py,
; user_data_dir) and only falls back to %LOCALAPPDATA% when it is not. Under
; Program Files it would always take the fallback, so the install a user can
; copy, inspect and delete stops being self-contained. %LOCALAPPDATA%\Programs
; keeps the portable behaviour and skips the UAC prompt.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Tag
  #define Tag AppVersion
#endif
#ifndef Repo
  #define Repo "Aseiel/VideoHighlighter"
#endif
; Number of .7z volumes on the release. CI counts the files it just produced.
#ifndef Volumes
  #define Volumes 2
#endif
; Payload size and unpacked size, for the free-space check. 0 disables it.
#ifndef ArchiveMB
  #define ArchiveMB 0
#endif
#ifndef InstalledMB
  #define InstalledMB 0
#endif

; Display name and install folder. Editions that can sit on the same machine
; must pass a different name AND a different AppId, or installing one takes the
; other's uninstall entry with it.
#ifndef AppName
  #define AppName "VideoHighlighter"
#endif
#ifndef AppId
  #define AppId "{6E1B9C34-4F27-4A88-9D0E-1C5A7B3F8E42}"
#endif
#ifndef OutputName
  #define OutputName "00-VideoHighlighter-Windows-Setup"
#endif

#define Publisher "Aseiel"
#define AppExe "VideoHighlighter.exe"
; Top-level folder inside the archive: both editions archive ./dist/VideoHighlighter,
; which is PyInstaller's --name, so this does not follow the display name.
#define PayloadRoot "VideoHighlighter"

; Overridable so the whole download-and-unpack path can be exercised against a
; local HTTP server and a small archive, instead of pulling gigabytes from a
; real release every time this script is touched.
#ifndef ArchiveBase
  #define ArchiveBase "VideoHighlighter-Windows-" + Tag + ".7z"
#endif
#ifndef DownloadBase
  #define DownloadBase "https://github.com/" + Repo + "/releases/download/" + Tag
#endif

; The file 7-Zip is pointed at. A split download starts from its .001 volume;
; an embedded archive is whatever CI handed us, unsplit.
#ifdef EmbeddedArchive
  #define PayloadName ExtractFileName(EmbeddedArchive)
#else
  #define PayloadName ArchiveBase + ".001"
#endif

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
AppPublisherURL=https://github.com/{#Repo}
WizardStyle=modern
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
OutputBaseFilename={#OutputName}
SetupIconFile=..\..\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Files]
; Unpacks the payload; extracted to {tmp} during install only. Carried rather
; than looked for because a machine with no 7-Zip is the machine this is for.
Source: "7zr.exe"; Flags: dontcopy
#ifdef EmbeddedArchive
; nocompression: it is a .7z already, and asking Inno to compress it again buys
; nothing and costs a long CI step on several gigabytes.
Source: "{#EmbeddedArchive}"; DestDir: "{tmp}"; Flags: nocompression deleteafterinstall
#endif

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式(&D)"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "启动 {#AppName}"; Flags: nowait postinstall skipifsilent

; Nothing under {app} is tracked by the installer -- 7-Zip wrote it -- so the
; uninstaller has to be told to take the folder. That includes what the app
; created at runtime: the analysis cache, debug.log, config.yaml, imported
; models. Uninstalling is therefore a full removal, which is what the entry in
; Apps & features promises.
[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
#ifndef EmbeddedArchive
var
  DownloadPage: TDownloadWizardPage;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax > 0 then
    DownloadPage.SetText(FileName,
      IntToStr(Progress div 1048576) + ' / ' + IntToStr(ProgressMax div 1048576) + ' MB');
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(
    '正在下载 {#AppName}',
    '正在从 GitHub 获取应用程序文件，下载体积较大，请保持网络连接。',
    @OnDownloadProgress);
end;

function VolumeName(Index: Integer): String;
begin
  Result := Format('%s.%.3d', ['{#ArchiveBase}', Index]);
end;
#endif

// Free space in MB on the volume holding Path, or -1 when it cannot be read.
// Asked about the drive root: {app} usually does not exist yet at this point.
function FreeSpaceMB(const Path: String): Int64;
var
  Free, Total: Int64;
begin
  if GetSpaceOnDisk64(AddBackslash(ExtractFileDrive(Path)), Free, Total) then
    Result := Free div 1048576
  else
    Result := -1;
end;

// The payload passes through {tmp} -- downloaded there, or copied there out of
// Setup.exe -- before it is unpacked into {app}, so both have to fit, and on a
// default install they are the same drive and need the sum. Checked up front
// rather than letting 7-Zip fail at the end of a multi-gigabyte transfer with a
// disk-full error that names neither number.
function ShortOfSpace(const Drive, What: String; Need, Free: Int64): Boolean;
begin
  Result := (Free >= 0) and (Free < Need);
  if Result then
    MsgBox('磁盘 ' + Drive + ' 可用空间不足（' + What + '）。' + #13#10#13#10
      + '需要：约 ' + IntToStr(Need) + ' MB' + #13#10
      + '可用：' + IntToStr(Free) + ' MB' + #13#10#13#10
      + '请释放磁盘空间，或返回上一步选择其他磁盘上的安装目录。',
      mbError, MB_OK);
end;

function SpaceLooksSufficient: Boolean;
var
  AppDir, TmpDir: String;
  NeedApp, NeedTmp: Int64;
begin
  Result := True;
  NeedApp := {#InstalledMB};
  NeedTmp := {#ArchiveMB};
  if (NeedApp <= 0) and (NeedTmp <= 0) then
    Exit;

  AppDir := ExpandConstant('{app}');
  TmpDir := ExpandConstant('{tmp}');

  if SameText(ExtractFileDrive(AppDir), ExtractFileDrive(TmpDir)) then
  begin
    Result := not ShortOfSpace(ExtractFileDrive(AppDir),
      '用于安装', NeedApp + NeedTmp, FreeSpaceMB(AppDir));
    Exit;
  end;

  if ShortOfSpace(ExtractFileDrive(AppDir), '用于安装',
                  NeedApp, FreeSpaceMB(AppDir)) then
  begin
    Result := False;
    Exit;
  end;

  // The temporary folder holds the payload until it is unpacked, so its drive
  // needs room for it even when the app is being installed elsewhere.
  if ShortOfSpace(ExtractFileDrive(TmpDir), '用于临时文件',
                  NeedTmp, FreeSpaceMB(TmpDir)) then
    Result := False;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
#ifndef EmbeddedArchive
var
  I: Integer;
#endif
begin
  if CurPageID <> wpReady then
  begin
    Result := True;
    Exit;
  end;

  if not SpaceLooksSufficient then
  begin
    Result := False;
    Exit;
  end;

#ifdef EmbeddedArchive
  Result := True;
#else
  DownloadPage.Clear;
  for I := 1 to {#Volumes} do
    DownloadPage.Add('{#DownloadBase}/' + VolumeName(I), VolumeName(I), '');

  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      Result := True;
    except
      // A dropped connection is the normal failure for a multi-gigabyte
      // download; leaving the wizard on this page lets Next retry it.
      if DownloadPage.AbortedByUser then
        Log('Download cancelled by the user.')
      else
        SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
      Result := False;
    end;
  finally
    DownloadPage.Hide;
  end;
#endif
end;

// 7-Zip wrote {app}\{#PayloadRoot}\* because the archive carries that top
// folder: CI archives ./dist/VideoHighlighter, and the portable download is
// meant to unpack into a named folder rather than scatter into the current one.
// Lift its contents one level so the exe is {app}\VideoHighlighter.exe instead
// of a repeated path. Same volume, so each move is a rename, whatever the size.
procedure LiftNestedFolder(const Nested: String);
var
  Rec: TFindRec;
  Names: TArrayOfString;
  I, N: Integer;
begin
  N := 0;
  if FindFirst(AddBackslash(Nested) + '*', Rec) then
  begin
    try
      repeat
        if (Rec.Name <> '.') and (Rec.Name <> '..') then
        begin
          SetArrayLength(Names, N + 1);
          Names[N] := Rec.Name;
          N := N + 1;
        end;
      until not FindNext(Rec);
    finally
      FindClose(Rec);
    end;
  end;

  for I := 0 to N - 1 do
    if not RenameFile(AddBackslash(Nested) + Names[I],
                      ExpandConstant('{app}\') + Names[I]) then
      Log('Could not move ' + Names[I] + ' out of ' + Nested);

  RemoveDir(Nested);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  SevenZip, Payload, Nested: String;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  ExtractTemporaryFile('7zr.exe');
  SevenZip := ExpandConstant('{tmp}\7zr.exe');
  Payload := ExpandConstant('{tmp}\{#PayloadName}');

  WizardForm.StatusLabel.Caption := '正在解压应用程序文件（可能需要几分钟）…';
  WizardForm.Refresh;

  // A split payload is entered through its .001 and 7-Zip finds the rest next
  // to it by name. Everything is in {tmp} -- put there by the download page or
  // copied out of Setup.exe -- and goes away with {tmp} when Setup exits.
  if not Exec(SevenZip, Format('x -y "%s" -o"%s"', [Payload, ExpandConstant('{app}')]),
              '', SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
  begin
    MsgBox('无法解压应用程序文件（7-Zip 退出代码：'
      + IntToStr(ResultCode) + '）。'#13#10#13#10
      + '安装未完成，请重新运行安装程序。', mbCriticalError, MB_OK);
    Exit;
  end;

  Nested := ExpandConstant('{app}\{#PayloadRoot}');
  if DirExists(Nested) then
    LiftNestedFolder(Nested);

  WizardForm.StatusLabel.Caption := '';
end;
