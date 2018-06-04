pipeline {
    agent {
        dockerfile {
            image 'Dockerfile.build'
            args '-v pip_cache:/var/pip_cache'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip install --cache-dir /var/pip_cache -r "requirements.txt"'
                sh 'pip install -e .'
                sh 'python qtt/test.py'
                sh 'py.test'
            }
        }
    }
}
