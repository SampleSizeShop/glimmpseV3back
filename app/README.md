#Build

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 543872078551.dkr.ecr.us-east-1.amazonaws.com

docker build -t glimmpsev3back .

docker tag glimmpsev3back:latest 543872078551.dkr.ecr.us-east-1.amazonaws.com/glimmpsev3back:0.0.27

docker push 543872078551.dkr.ecr.us-east-1.amazonaws.com/glimmpsev3back:0.0.27

# Deploy
