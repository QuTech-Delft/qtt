pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }
    stages {
        stage('Install qcodes')
        {
            steps {
                sh 'ls -als'
                sh 'cd Qcodes'
                sh 'python3 setup.py install --user'
                sh 'python3 setup.py build'
                sh 'python -c "import qcodes"'
            }
        }
    }
}
