pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'python --version'
                sh 'ls -lah'
                sh 'pip3 install --cache-dir /var/pip_cache -r "requirements_linux.txt"'
                sh 'pip3 install .'
                sh 'python3 qtt/test.py'
                sh 'py.test'
            }
        }
    }
}
