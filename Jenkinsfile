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
                sh 'apt update'
                sh 'apt install libgl1-mesa-glx'
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip install --cache-dir /var/pip_cache -r "requirements_linux.txt"'
                sh 'pip install .'
                sh 'py.test'
                sh 'python qtt/test.py'
            }
        }
    }
}
