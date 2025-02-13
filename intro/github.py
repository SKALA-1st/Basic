import sys
import requests

def get_github_user_info(username):
    """
    GitHub API를 사용하여 특정 사용자의 정보를 가져오는 함수
    """
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_github_repo_info(username, repo_name):
    """
    GitHub API를 사용하여 특정 사용자의 특정 레포지토리 정보를 가져오는 함수
    """
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    if len(sys.argv) != 3:
        print("사용법: python github_user_info.py <GitHub_Username> <Repository_Name>")
        sys.exit(1)
    
    username = sys.argv[1]
    repo_name = sys.argv[2]
    user_info = get_github_user_info(username)
    repo_info = get_github_repo_info(username, repo_name)
    
    if user_info:
        print(f"사용자: {user_info['login']}")
        print(f"이름: {user_info.get('name', '정보 없음')}")
        print(f"회사: {user_info.get('company', '정보 없음')}")
        print(f"위치: {user_info.get('location', '정보 없음')}")
        print(f"공개 저장소 수: {user_info.get('public_repos', 0)}")
    else:
        print("사용자 정보를 가져올 수 없습니다.")
        sys.exit(1)
    
    if repo_info:
        print(f"레포지토리: {repo_info['name']}")
        print(f"설명: {repo_info.get('description', '정보 없음')}")
        print(f"스타 수: {repo_info.get('stargazers_count', 0)}")
        print(f"포크 수: {repo_info.get('forks_count', 0)}")
    else:
        print("레포지토리 정보를 가져올 수 없습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
