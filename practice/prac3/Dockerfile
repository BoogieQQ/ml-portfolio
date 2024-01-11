FROM python:3.12-slim

COPY src/requirements.txt /root/prac/src/requirements.txt

RUN chown -R root:root /root/prac

WORKDIR /root/prac/src
RUN pip3 install -r requirements.txt

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY my_secret_key
ENV FLASK_APP server

RUN chmod +x run.py
CMD ["python3", "run.py"]