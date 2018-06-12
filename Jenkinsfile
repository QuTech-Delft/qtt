pipeline {
    agent {
        docker {
            image 'continuumio/anaconda3'
            args '-v pip_cache:/var/pip_cache'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'apt-get update && apt-get install libgl1-mesa-glx libx11-xcb1 -y'
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip install --cache-dir /var/pip_cache -r "requirements_linux.txt"'
                sh 'pip install .'
                sh 'python qtt/test.py'
                sh 'py.test'
            }
        }
    }
}
