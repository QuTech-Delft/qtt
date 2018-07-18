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
                sh 'pip3 install -r requirements.txt'
                sh 'python3 setup.py build'
                sh 'python3 setup.py install --user'
            }
        }
    }
}
