# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /first_mlops

# Copy the current directory contents into the container at /app
COPY . /first_mlops

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "iris_classification.py"]