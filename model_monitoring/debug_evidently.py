import sys
try:
    import evidently
    print(f"✅ Evidently found at: {evidently.__file__}")
    print(f"✅ Version: {evidently.__version__}")
    
    from evidently.report import Report
    print("✅ Successfully imported evidently.report.Report")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Python executable:", sys.executable)