# conda create -n python38 python=3.8 -y
# conda activate python38
pip install -r requirements.txt --target python
zip -q -r python.zip python
