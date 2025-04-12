pip3 install -U pip
pip3 install -r ../requirements.txt
python3 -m pip install --no-cache-dir scipy==1.1.0
pip3 install --upgrade 'sentry-sdk[flask]'
pip3 install pandas
pip3 install openpyxl
python3 Validation_to_latex.py 