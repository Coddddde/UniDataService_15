FROM python

WORKDIR ./docker_demo

COPY . .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

CMD ["python","api_rec.py"]