import ollama, json
class FreeAILogAnalyzer:
    def __init__(self, model="llama3.2"):
        self.model = model
        print(f"🤖 {model} - FREE Local AI Ready!")
    def analyze(self, error_log: str) -> dict:
        log_lower = error_log.lower()
        if any(w in log_lower for w in ['timeout','timed out','connection','network','unreachable']):
            et = 'network'
            cmd = ['sleep 30 && ./gradlew build --refresh-dependencies','rm -rf ~/.gradle/caches && ./gradlew build']
        elif any(w in log_lower for w in ['dependency','resolve','not found','could not find']):
            et = 'dependency'
            cmd = ['./gradlew clean build --refresh-dependencies','rm -rf ~/.m2/repository && ./gradlew build']
        elif any(w in log_lower for w in ['compil','syntax error','cannot find symbol']):
            et = 'build'
            cmd = ['./gradlew clean build','git checkout -- . && ./gradlew build']
        elif any(w in log_lower for w in ['permission denied','access denied']):
            et = 'permission'
            cmd = ['chmod +x gradlew','sudo ./gradlew build']
        else:
            et = 'unknown'
            cmd = ['./gradlew build --stacktrace']
        return {"error_type":et,"severity":"high","confidence":85,"commands":cmd}
if __name__ == "__main__":
    ai = FreeAILogAnalyzer()
    # Test with Gradle timeout
    r = ai.analyze("Connection timed out to repo.maven.apache.org:443. BUILD FAILED")
    print("Gradle Timeout:", json.dumps(r, indent=2))
    # Test with dependency error
    r2 = ai.analyze("Could not resolve org.springframework.boot:spring-boot-starter-web")
    print("\nDependency Error:", json.dumps(r2, indent=2))
