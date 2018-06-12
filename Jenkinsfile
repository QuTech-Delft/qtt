pipeline {
    agent {
        docker {
            image 'python:3.6'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip install -r "requirements_linux.txt"'
                sh 'pip install .'
                sh 'python qtt/test.py'
                sh 'py.test'
            }
        }
    }
}
