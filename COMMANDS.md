Directory:/home/lily/ch956/chimera
Cache:
	/home/lily/ch956/chimera/cache/WebNLG
	/home/lily/ch956/chimera/ .TEMP_OLD
.TEMP is For trained models
Renamed as .TEMP_OLD, change back to .TEMP if you want to use this

Commands(after cd into directory):
	       source chienv/bin/activate
Naive Planner:
      python server/server.py
Make sure in the server.py config dictionary “low_mem” is set to True
This planner consumes lots of memory and gets killed during test set preprocessing(around 900 ish). I did a workaround by saving every preprocessed test set so that it picks up from where it was killed. If you’re running this you might need to rerun 2-3 times to finish preprocessing the whole test set.

Neural Planner:
       python main.py
Note 1: switching between naive and neural planner can be done by changing the “planner” value in the config dictionary. Currently if you execute main.py, the planner is already set to neural planner, and for server.py it’s already set to naive planner. Of course you can also just modify the config in the server.py file.
Note 2: If you previously ran one of the planners, make sure to clear cache before running another one(in cache/WebNLG/ remove the  ‘train-planner’ and possibly “test-corpus”, “train-reg”, “translate”, and “evaluate” if exists.)