@echo off
rem Build a small, very clean training set for the Stage-1 smoke test, from the mined dataset.
rem Usage (in the punarvaani-data folder, conda env active):   prepare_smoke_set.bat [hours]
rem Safe while `pvdata auto` is running: it only reads dataset\ and writes smoke_v1\ + a zip.
setlocal
set HOURS=%1
if "%HOURS%"=="" set HOURS=1.6
set OUT=smoke_v1
if exist %OUT% (
  echo %OUT% already exists. Rename or delete it first, then run this again.
  exit /b 1
)
echo [1/5] Picking the best-scored clips, balanced across people (%HOURS% h before clean-up)...
pvdata snapshot --out dataset --to %OUT% --hours %HOURS% --gold || goto :fail
echo [2/5] Face identity: grouping videos into people, removing other faces (a few minutes)...
pvdata identify --out %OUT% --quarantine || goto :fail
echo [3/5] Person-disjoint train / val / test split...
pvdata split --out %OUT% || goto :fail
echo [4/5] Checking every clip against the trainer's rules...
pvdata validate %OUT%\raw_videos || goto :fail
pvdata report --out %OUT% > %OUT%\report_summary.txt
echo [5/5] Zipping...
pushd %OUT%
tar -a -c -f ..\punarvaani_smoke_v1.zip raw_videos manifest.csv report.json speakers.json splits.json identity || (popd & goto :fail)
popd
echo.
echo Done: punarvaani_smoke_v1.zip   (summary in %OUT%\report_summary.txt)
findstr /C:"accepted_hours" /C:"\"accepted\"" /C:"\"speakers\"" %OUT%\report_summary.txt
exit /b 0
:fail
echo.
echo FAILED at the step above. Send the messages above to Kartikey.
exit /b 1
