To build:
```docker build . -t mksim`
```
To run:
```docker run -it --rm -d -v "path to local data vol":/scratch mksim:latest /bin/bash
docker exec -it "CONTAINERID" /bin/bash
tmux; cd /scratch
python /home/kat/sim/mk+sim/obit_sim_mk+.py
```
