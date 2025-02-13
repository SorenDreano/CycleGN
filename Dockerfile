FROM nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:ae8a022c02aec945c4f8c52f65deaf535de7abb58e840350d19391ec683f4980

WORKDIR /app
COPY requirements.txt .
COPY init.py .
COPY train.py .
COPY cyclegn_utils.py .
RUN chmod +x init.py
RUN chmod +x train.py
RUN apt-get update
RUN apt-get install -y python3.11=3.11.0~rc1-1~22.04 python3-pip=22.0.2+dfsg-1ubuntu0.5
RUN rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["bash", "-c"]
# Example: docker run --rm -it --gpus all -v ./:/app cyclegn /app/init.py --X_train eng.txt --Y_train fra.txt