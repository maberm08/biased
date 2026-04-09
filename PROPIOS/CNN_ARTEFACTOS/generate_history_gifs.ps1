param(
    [Parameter(Mandatory = $true)]
    [string[]]$HistoryPaths,
    [Parameter(Mandatory = $true)]
    [string]$LossGifPath,
    [Parameter(Mandatory = $true)]
    [string]$AccuracyGifPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName PresentationCore

$ModelNames = @{
    "cnn" = "CNN"
    "mlp" = "MLP"
    "kan" = "KAN"
}

$ModelColors = @{
    "cnn" = [System.Drawing.Color]::FromArgb(31, 119, 180)
    "mlp" = [System.Drawing.Color]::FromArgb(214, 39, 40)
    "kan" = [System.Drawing.Color]::FromArgb(44, 160, 44)
}

function Load-Histories {
    param([string[]]$Paths)

    $histories = @{}
    foreach ($path in $Paths) {
        $file = Get-Item $path
        $modelKey = $file.BaseName.Split("_")[0].ToLowerInvariant()
        if (-not $ModelNames.ContainsKey($modelKey)) {
            throw "No se pudo inferir el modelo desde $($file.Name)"
        }
        $histories[$modelKey] = Get-Content $file.FullName -Raw | ConvertFrom-Json
    }
    return $histories
}

function ConvertTo-BitmapFrame {
    param([System.Drawing.Bitmap]$Bitmap)

    $memoryStream = New-Object System.IO.MemoryStream
    $Bitmap.Save($memoryStream, [System.Drawing.Imaging.ImageFormat]::Png)
    $null = $memoryStream.Seek(0, [System.IO.SeekOrigin]::Begin)
    $decoder = [System.Windows.Media.Imaging.BitmapDecoder]::Create(
        $memoryStream,
        [System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
        [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    )
    $frame = $decoder.Frames[0]
    $memoryStream.Close()
    return $frame
}

function Draw-LineSeries {
    param(
        [System.Drawing.Graphics]$Graphics,
        [double[]]$Values,
        [int]$EpochCount,
        [double]$YMax,
        [int]$PlotLeft,
        [int]$PlotTop,
        [int]$PlotWidth,
        [int]$PlotHeight,
        [System.Drawing.Pen]$Pen
    )

    if ($Values.Length -eq 0) {
        return
    }

    $points = New-Object 'System.Collections.Generic.List[System.Drawing.PointF]'
    for ($i = 0; $i -lt $Values.Length; $i++) {
        if ($EpochCount -le 1) {
            $x = [single]($PlotLeft + ($PlotWidth / 2.0))
        } else {
            $x = [single]($PlotLeft + ($i / ($EpochCount - 1.0)) * $PlotWidth)
        }
        if ($YMax -le 0) {
            $y = [single]($PlotTop + $PlotHeight)
        } else {
            $y = [single]($PlotTop + $PlotHeight - (($Values[$i] / $YMax) * $PlotHeight))
        }
        $points.Add([System.Drawing.PointF]::new($x, $y))
    }

    if ($points.Count -eq 1) {
        $point = $points[0]
        $Graphics.FillEllipse([System.Drawing.Brushes]::Black, $point.X - 2, $point.Y - 2, 4, 4)
        return
    }

    $Graphics.DrawLines($Pen, $points.ToArray())
}

function Create-MetricGif {
    param(
        [hashtable]$HistoriesByModel,
        [string]$OutputPath,
        [string]$TrainKey,
        [string]$ValKey,
        [string]$YAxisLabel,
        [string]$Title
    )

    $width = 900
    $height = 560
    $plotLeft = 80
    $plotTop = 60
    $plotWidth = 760
    $plotHeight = 390
    $plotBottom = $plotTop + $plotHeight
    $plotRight = $plotLeft + $plotWidth

    $fontTitle = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)
    $fontAxis = New-Object System.Drawing.Font("Arial", 10)
    $fontLegend = New-Object System.Drawing.Font("Arial", 9)

    $maxEpochs = 0
    $maxMetric = 0.0
    foreach ($history in $HistoriesByModel.Values) {
        if ($history.Count -gt $maxEpochs) {
            $maxEpochs = $history.Count
        }
        foreach ($entry in $history) {
            $trainValue = [double]$entry.$TrainKey
            $valValue = [double]$entry.$ValKey
            if ($trainValue -gt $maxMetric) { $maxMetric = $trainValue }
            if ($valValue -gt $maxMetric) { $maxMetric = $valValue }
        }
    }
    if ($YAxisLabel -eq "Accuracy") {
        $maxMetric = [Math]::Min(1.0, [Math]::Max($maxMetric * 1.05, 0.1))
    } else {
        $maxMetric = [Math]::Max($maxMetric * 1.1, 0.1)
    }

    $encoder = New-Object System.Windows.Media.Imaging.GifBitmapEncoder

    for ($epoch = 1; $epoch -le $maxEpochs; $epoch++) {
        $bitmap = New-Object System.Drawing.Bitmap($width, $height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.Clear([System.Drawing.Color]::White)

        $graphics.DrawString("$Title - epoca $epoch", $fontTitle, [System.Drawing.Brushes]::Black, 25, 18)

        $axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::Black, 1.4)
        $gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray, 1.0)
        $graphics.DrawRectangle($axisPen, $plotLeft, $plotTop, $plotWidth, $plotHeight)

        for ($g = 1; $g -lt 5; $g++) {
            $gridY = [int]($plotTop + ($g / 5.0) * $plotHeight)
            $graphics.DrawLine($gridPen, $plotLeft, $gridY, $plotRight, $gridY)
        }

        foreach ($modelKey in @("cnn", "mlp", "kan")) {
            if (-not $HistoriesByModel.ContainsKey($modelKey)) {
                continue
            }

            $history = $HistoriesByModel[$modelKey]
            $current = @($history | Select-Object -First ([Math]::Min($epoch, $history.Count)))
            $trainValues = @($current | ForEach-Object { [double]$_.$TrainKey })
            $valValues = @($current | ForEach-Object { [double]$_.$ValKey })
            $color = $ModelColors[$modelKey]

            $trainPen = New-Object System.Drawing.Pen($color, 3.0)
            $valPen = New-Object System.Drawing.Pen($color, 2.0)
            $valPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

            Draw-LineSeries -Graphics $graphics -Values $trainValues -EpochCount $maxEpochs -YMax $maxMetric -PlotLeft $plotLeft -PlotTop $plotTop -PlotWidth $plotWidth -PlotHeight $plotHeight -Pen $trainPen
            Draw-LineSeries -Graphics $graphics -Values $valValues -EpochCount $maxEpochs -YMax $maxMetric -PlotLeft $plotLeft -PlotTop $plotTop -PlotWidth $plotWidth -PlotHeight $plotHeight -Pen $valPen

            $trainPen.Dispose()
            $valPen.Dispose()
        }

        $graphics.DrawString("Epoca", $fontAxis, [System.Drawing.Brushes]::Black, [single]($plotLeft + ($plotWidth / 2) - 20), [single]($plotBottom + 18))
        $graphics.DrawString($YAxisLabel, $fontAxis, [System.Drawing.Brushes]::Black, 18, [single]($plotTop + ($plotHeight / 2)))

        $legendX = $plotLeft
        $legendY = $plotBottom + 42
        foreach ($modelKey in @("cnn", "mlp", "kan")) {
            if (-not $HistoriesByModel.ContainsKey($modelKey)) {
                continue
            }

            $color = $ModelColors[$modelKey]
            $brush = New-Object System.Drawing.SolidBrush($color)
            $graphics.FillRectangle($brush, $legendX, $legendY + 5, 16, 8)
            $graphics.DrawString("$($ModelNames[$modelKey]) train / val", $fontLegend, [System.Drawing.Brushes]::Black, [single]($legendX + 22), [single]$legendY)
            $legendX += 180
            $brush.Dispose()
        }

        $frame = ConvertTo-BitmapFrame -Bitmap $bitmap
        for ($repeat = 0; $repeat -lt 5; $repeat++) {
            $encoder.Frames.Add($frame)
        }

        $gridPen.Dispose()
        $axisPen.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }

    $fontLegend.Dispose()
    $fontAxis.Dispose()
    $fontTitle.Dispose()

    $outputDir = Split-Path -Parent $OutputPath
    if ($outputDir -and -not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    }
    $fileStream = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create)
    $encoder.Save($fileStream)
    $fileStream.Close()
}

$histories = Load-Histories -Paths $HistoryPaths

Create-MetricGif -HistoriesByModel $histories -OutputPath $LossGifPath -TrainKey "train_loss" -ValKey "val_loss" -YAxisLabel "Loss" -Title "Curvas de aprendizaje CNN / MLP / KAN"
Create-MetricGif -HistoriesByModel $histories -OutputPath $AccuracyGifPath -TrainKey "train_acc" -ValKey "val_acc" -YAxisLabel "Accuracy" -Title "Curvas de accuracy CNN / MLP / KAN"

Write-Output "Loss GIF: $LossGifPath"
Write-Output "Accuracy GIF: $AccuracyGifPath"
