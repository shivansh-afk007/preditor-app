# Activate virtual environment
.\.venv\Scripts\activate

# Install AWS CLI if not installed
if (!(Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "Installing AWS CLI..."
    pip install awscli
}

# Install EB CLI if not installed
if (!(Get-Command eb -ErrorAction SilentlyContinue)) {
    Write-Host "Installing EB CLI..."
    pip install awsebcli
}

# Create deployment package
Write-Host "Creating deployment package..."
$excludeDirs = @('.git', '.venv', 'venv', '__pycache__', 'instance', 'migrations')
$excludeFiles = @('*.pyc', '*.pyo', '*.pyd', '*.db', '*.sqlite3', '*.csv', '*.ipynb', '*.pdf', '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.mov', '*.mp4', '*.mp3', '*.wav', '*.zip', '*.tar.gz', '*.rar', '*.7z', '*.log', '*.tmp', '*.temp', '*.swp', '*.swo', '*.bak', '*.backup', '*.orig', '*.rej', '*.diff', '*.patch', '*.DS_Store', 'Thumbs.db')

# Get all files excluding specified directories and files
$files = Get-ChildItem -Path . -Recurse -File | Where-Object {
    $file = $_
    $dir = $file.DirectoryName
    -not ($excludeDirs | Where-Object { $dir -like "*\$_*" }) -and
    -not ($excludeFiles | Where-Object { $file.Name -like $_ })
}

# Create zip file
$zipPath = "deploy.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(".", $zipPath)

# Deploy to Elastic Beanstalk
Write-Host "Deploying to Elastic Beanstalk..."
eb deploy

# Clean up
Write-Host "Cleaning up..."
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Write-Host "Deployment completed!" 