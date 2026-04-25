Run python src/evaluate.py
Traceback (most recent call last):
  File "/home/runner/work/mlip-walmart-forecast/mlip-walmart-forecast/src/evaluate.py", line 140, in <module>
    main()
  File "/home/runner/work/mlip-walmart-forecast/mlip-walmart-forecast/src/evaluate.py", line 127, in main
    validate_inputs(test_df, forecasts_df)
  File "/home/runner/work/mlip-walmart-forecast/mlip-walmart-forecast/src/evaluate.py", line 31, in validate_inputs
    raise ValueError(f"Faltan columnas en forecasts: {missing_fcst}")
ValueError: Faltan columnas en forecasts: ['q10', 'q50', 'q90']
Error: Process completed with exit code 1.