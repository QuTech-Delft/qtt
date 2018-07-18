pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }
    stages {
        stage('Import')
        {
            steps {
                sh 'python3 -c "import matplotlib.pyplot as plt"'
                sh 'python3 -c "import scipy"'
                sh 'python3 -c "import qtt"'
            }
        }
        stage('Test') {
            steps {
                sh 'python3 --version'
                sh 'ls -lah'
                sh 'py.test-3 -k qtt --ignore qtt/legacy.py'
            }
        }
    }
}
