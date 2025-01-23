FROM python:3.12
WORKDIR /app

# Install the application dependencies
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install --no-cache-dir -r Requirements.txt

#need to install at last due to pip not being able to install torchvision before detectro
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git 

#use --no-build-isolation if "no module named torch found" error appears.

CMD ["python3", "app.py"]