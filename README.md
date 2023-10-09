# temp

Prerequisites:
Your computer is installed Python

Clone source code from git. 
```bash
git clone https://github.com/Clarence161095/temp.git
```

Direct to temp project
```bash
cd temp
```

Install library
```bash
pip3 install Flask scikit-learn pandas tensorflow-cpu
```

Open port
```bash
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
```

(Optional) Training model
```bash
python3 training.py
```

Step 4: Serve app
```bash
python3 app.py
```


access: 
http://localhost/
or
IP
