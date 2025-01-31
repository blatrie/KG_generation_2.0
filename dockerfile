# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /KG_generation_2.0

# Copy the local code to the container
COPY . /KG_generation_2.0

RUN pip3 install -U sentence-transformers

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to src/
WORKDIR /KG_generation_2.0/src

# Define environment variable
ENV NAME KG-Gen

# Run streamlit when the container launches
CMD ["streamlit", "run", "main.py"]