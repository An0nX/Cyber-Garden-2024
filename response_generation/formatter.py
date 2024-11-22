import csv
from io import StringIO

def format_response(analysis_result):
    if "symptoms" in analysis_result and "diagnosis" in analysis_result:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Symptoms", "Diagnosis"])
