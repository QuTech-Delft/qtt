pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile.build'
        }
    }
    stages {
        stage('Install')
        {
            steps {
                sh 'mkdir -p git'
                sh ' cd git'
                sh 'git clone git@github.com:VandersypenQutech/QCoDeS.git'
                sh 'git clone git@github.com:VandersypenQutech/qtt.git'

                sh 'cd git/QCoDeS'
                sh 'pip3 install -r requirements.txt'
                sh 'python3 setup.py build'
                sh 'python3 setup.py install --user'
            }
        }
    }
}
