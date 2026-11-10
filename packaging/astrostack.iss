; Script Inno Setup : crée l'installeur Windows AstroStack-Setup-<version>.exe
; Compilé par la CI avec : ISCC /DAppVersion=<version> packaging\astrostack.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppName=AstroStack
AppVersion={#AppVersion}
AppPublisher=David Sanchez
DefaultDirName={autopf}\AstroStack
DefaultGroupName=AstroStack
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=AstroStack-Setup-{#AppVersion}
SetupIconFile=build\icon.ico
UninstallDisplayIcon={app}\AstroStack.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Créer une icône sur le Bureau / Create a desktop icon"; GroupDescription: "Raccourcis / Shortcuts:"

[Files]
Source: "..\dist\AstroStack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AstroStack"; Filename: "{app}\AstroStack.exe"
Name: "{autodesktop}\AstroStack"; Filename: "{app}\AstroStack.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AstroStack.exe"; Description: "Lancer AstroStack / Launch AstroStack"; Flags: nowait postinstall skipifsilent
