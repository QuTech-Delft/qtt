pipeline {
    agent {
        docker {
            image 'jfloff/alpine-python'
            args '-v pip_cache:/var/pip_cache'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
                sh 'pwd'
                sh 'ls -lah'
            }
        }
    }
} 
