#Run Validation Suite
To run the validation code for this application:

 - Clone this repository.
 - ensure you have python3 installed.
 - We highly recommend creating a virtualenv or conda environment.
 - run the shell script GlimmpseValidation.sh, found in the app directory.

#Build
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 543872078551.dkr.ecr.us-east-1.amazonaws.com

docker build -t glimmpsev3back .

docker tag glimmpsev3back:latest 543872078551.dkr.ecr.us-east-1.amazonaws.com/glimmpsev3back:0.0.27

docker push 543872078551.dkr.ecr.us-east-1.amazonaws.com/glimmpsev3back:0.0.27

# Deploy
