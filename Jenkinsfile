pipeline {
    agent {
        docker {
            image 'miniconda3'
            args '-v pip_cache:/var/pip_cache'
        }
    }
    stages {
        stage('Test') {
            steps {
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
