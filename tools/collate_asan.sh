#!/bin/bash
# Collate all ASAN logs
# Make sure to run all executables with the environmental variable
# ASAN_OPTIONS="log_path=ASANlogger"

OUTDIR="Testing/Temporary/"
OUTFILE="${OUTDIR}/LastTest_ASAN.log"
HTML="${OUTDIR}/LastTest_ASAN.html"

mkdir -p $OUTDIR
echo "COLLATED ASAN LOG" >  $OUTFILE
LOGS=`find . -name 'ASANlogger.*'`

if [ -z "$LOGS" ] ; then
  echo "NO LOGS FOUND?" | tee $OUTFILE
  echo '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html" />
<title>Collated ASAN Log Report</title>
</head>
<body>
<h1>Collated AddressSanitizer (ASAN) Log Report</h1>
' > $HTML
date >> $HTML
echo ' <br>
NO LOG FILES FOUND, did the job complete successfully?
</body></html>' >> $HTML
  exit 0
fi

cat $LOGS >> $OUTFILE

echo '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html" />
<title>Collated ASAN Log Report</title>
<style> html,body,h1,h2,h3 {font-family: "Roboto", sans-serif;} </style>
</head>
<body>
<h1>Collated AddressSanitizer (ASAN) Log Report</h1>' > $HTML
date >> $HTML
echo '<h2>Summary</h2>
<table>
<tr>
<th>Number of logs found</th> <th> ' >> $HTML
echo $LOGS | wc -w >> $HTML
echo '</th></tr><tr></tr>' >> $HTML
for L in $LOGS ; do
  echo '<tr><td><code>'
  echo $L '</code></td><td><b>'
  grep -s --color=never '^SUMMARY:' $L | sed 's/^.*://'
  echo '</b></td></tr>'
done >> $HTML
echo '</table>
<h2>Incidents</h2>
' >> $HTML
echo -e '\n\n' >> $HTML
for L in $LOGS ; do
  echo '<h2>' $L '</h2>' >> $HTML
  echo '<code>' >> $HTML
  cat $L | sed 's/$/<br>/' >> $HTML
  echo -e '</code>\n\n' >> $HTML
done

echo '</body></html>' >> $HTML

