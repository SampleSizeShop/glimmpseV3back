FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY ./app /app
COPY requirements.txt /tmp/
ENV UWSGI_INI /app/uwsgi.ini
WORKDIR /app

RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir scipy==1.1.0
RUN python3 -m pip install --no-cache-dir --index-url https://test.pypi.org/simple/ pyglimmpse==0.0.21


ENV LISTEN_PORT 5000
EXPOSE 5000
ENV FLASK_APP=app
ENV FLASK_DEBUG=true

