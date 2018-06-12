pipeline {
    agent {
        docker {
            image 'python:3.6'
            args '-v pip_cache:/var/pip_cache'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip3 install --cache-dir /var/pip_cache -r "requirements_linux.txt"'
                sh 'python qtt/test.py'
                sh 'py.test'
            }
        }
    }
}
