#!/usr/bin/env python3
"""
Synthetic Data Generator for CodeFault Experiments

Generates realistic code change + log context pairs for training and evaluation.
This enables running experiments without external data dependencies.

Bug patterns based on:
- ManySStuBs4J bug taxonomy
- Common Java/Python exceptions
- Real-world fault scenarios
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


# Bug-inducing code change templates
BUG_PATTERNS = {
    'null_pointer': {
        'diff_templates': [
            """--- a/src/{project}/service/UserService.java
+++ b/src/{project}/service/UserService.java
@@ -42,7 +42,6 @@ public class UserService {{
     public String getUserName(User user) {{
-        if (user != null && user.getProfile() != null) {{
-            return user.getProfile().getName();
-        }}
-        return "Unknown";
+        return user.getProfile().getName();
     }}
 }}""",
            """--- a/src/{project}/handler/RequestHandler.java
+++ b/src/{project}/handler/RequestHandler.java
@@ -89,8 +89,5 @@ public class RequestHandler {{
     public Response process(Request request) {{
-        if (request.getData() != null) {{
-            return transform(request.getData());
-        }}
-        return Response.empty();
+        return transform(request.getData());
     }}
 }}""",
            """--- a/src/{project}/util/ConfigLoader.java
+++ b/src/{project}/util/ConfigLoader.java
@@ -23,5 +23,4 @@ public class ConfigLoader {{
     public String getValue(String key) {{
-        Object value = config.get(key);
-        return value != null ? value.toString() : "";
+        return config.get(key).toString();
     }}
 }}""",
        ],
        'log_templates': [
            ["ERROR [UserService] Caused by: java.lang.NullPointerException",
             "at UserService.getUserName(UserService.java:42)",
             "at RequestHandler.handle(RequestHandler.java:156)"],
            ["FATAL [RequestHandler] NullPointerException in process()",
             "Request data was unexpectedly null",
             "Stack trace: RequestHandler.process:89"],
            ["ERROR [ConfigLoader] getValue() failed",
             "java.lang.NullPointerException: Cannot invoke toString() on null",
             "at ConfigLoader.getValue(ConfigLoader.java:23)"],
        ]
    },
    'array_index': {
        'diff_templates': [
            """--- a/src/{project}/util/ArrayUtils.java
+++ b/src/{project}/util/ArrayUtils.java
@@ -15,8 +15,6 @@ public class ArrayUtils {{
     public static int getLast(int[] arr) {{
-        if (arr != null && arr.length > 0) {{
-            return arr[arr.length - 1];
-        }}
-        return -1;
+        return arr[arr.length];
     }}
 }}""",
            """--- a/src/{project}/data/DataProcessor.java
+++ b/src/{project}/data/DataProcessor.java
@@ -67,7 +67,5 @@ public class DataProcessor {{
     public Object getElement(List<Object> list, int index) {{
-        if (index >= 0 && index < list.size()) {{
-            return list.get(index);
-        }}
-        return null;
+        return list.get(index);
     }}
 }}""",
        ],
        'log_templates': [
            ["ERROR [ArrayUtils] java.lang.ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 5",
             "at ArrayUtils.getLast(ArrayUtils.java:15)"],
            ["FATAL [DataProcessor] IndexOutOfBoundsException",
             "Index: 10, Size: 8",
             "at DataProcessor.getElement(DataProcessor.java:67)"],
        ]
    },
    'wrong_operator': {
        'diff_templates': [
            """--- a/src/{project}/math/Calculator.java
+++ b/src/{project}/math/Calculator.java
@@ -31,5 +31,5 @@ public class Calculator {{
     public int compare(int a, int b) {{
-        return a < b ? -1 : (a > b ? 1 : 0);
+        return a < b ? -1 : (a < b ? 1 : 0);
     }}
 }}""",
            """--- a/src/{project}/validator/RangeValidator.java
+++ b/src/{project}/validator/RangeValidator.java
@@ -12,5 +12,5 @@ public class RangeValidator {{
     public boolean isInRange(int value, int min, int max) {{
-        return value >= min && value <= max;
+        return value >= min || value <= max;
     }}
 }}""",
            """--- a/src/{project}/loop/Iterator.java
+++ b/src/{project}/loop/Iterator.java
@@ -45,5 +45,5 @@ public class Iterator {{
     public void process(int count) {{
-        for (int i = 0; i < count; i++) {{
+        for (int i = 0; i <= count; i++) {{
             doWork(i);
         }}
     }}
 }}""",
        ],
        'log_templates': [
            ["WARN [Calculator] Unexpected comparison result",
             "Expected: -1, Got: 0 for values a=5, b=10",
             "Test assertion failed"],
            ["ERROR [RangeValidator] Validation logic error",
             "Value 100 incorrectly validated as in range [0, 50]",
             "Bug in isInRange() method"],
            ["ERROR [Iterator] ArrayIndexOutOfBoundsException",
             "Off-by-one error in loop iteration",
             "at Iterator.process(Iterator.java:45)"],
        ]
    },
    'resource_leak': {
        'diff_templates': [
            """--- a/src/{project}/io/FileHandler.java
+++ b/src/{project}/io/FileHandler.java
@@ -28,12 +28,8 @@ public class FileHandler {{
     public String readFile(String path) throws IOException {{
-        BufferedReader reader = null;
-        try {{
-            reader = new BufferedReader(new FileReader(path));
-            return reader.readLine();
-        }} finally {{
-            if (reader != null) reader.close();
-        }}
+        BufferedReader reader = new BufferedReader(new FileReader(path));
+        return reader.readLine();
     }}
 }}""",
            """--- a/src/{project}/db/ConnectionPool.java
+++ b/src/{project}/db/ConnectionPool.java
@@ -55,10 +55,6 @@ public class ConnectionPool {{
     public ResultSet query(String sql) {{
         Connection conn = getConnection();
-        try {{
-            return conn.prepareStatement(sql).executeQuery();
-        }} finally {{
-            releaseConnection(conn);
-        }}
+        return conn.prepareStatement(sql).executeQuery();
     }}
 }}""",
        ],
        'log_templates': [
            ["WARN [FileHandler] Too many open files",
             "java.io.IOException: Too many open files in system",
             "Resource leak detected in readFile()"],
            ["ERROR [ConnectionPool] Connection pool exhausted",
             "No available connections after 30s timeout",
             "Possible connection leak in query()"],
        ]
    },
    'type_error': {
        'diff_templates': [
            """--- a/src/{project}/parser/JsonParser.java
+++ b/src/{project}/parser/JsonParser.java
@@ -19,5 +19,5 @@ public class JsonParser {{
     public int parseNumber(Object value) {{
-        return Integer.parseInt(String.valueOf(value));
+        return (Integer) value;
     }}
 }}""",
            """--- a/src/{project}/converter/TypeConverter.java
+++ b/src/{project}/converter/TypeConverter.java
@@ -33,7 +33,5 @@ public class TypeConverter {{
     public List<String> toStringList(Object obj) {{
-        if (obj instanceof List) {{
-            return ((List<?>) obj).stream().map(String::valueOf).collect(Collectors.toList());
-        }}
-        return Collections.singletonList(String.valueOf(obj));
+        return (List<String>) obj;
     }}
 }}""",
        ],
        'log_templates': [
            ["ERROR [JsonParser] java.lang.ClassCastException",
             "Cannot cast java.lang.String to java.lang.Integer",
             "at JsonParser.parseNumber(JsonParser.java:19)"],
            ["FATAL [TypeConverter] ClassCastException in toStringList()",
             "java.util.HashMap cannot be cast to java.util.List",
             "at TypeConverter.toStringList(TypeConverter.java:33)"],
        ]
    },
    'concurrency': {
        'diff_templates': [
            """--- a/src/{project}/cache/CacheManager.java
+++ b/src/{project}/cache/CacheManager.java
@@ -42,9 +42,7 @@ public class CacheManager {{
-    private final Object lock = new Object();

     public void put(String key, Object value) {{
-        synchronized(lock) {{
-            cache.put(key, value);
-        }}
+        cache.put(key, value);
     }}
 }}""",
            """--- a/src/{project}/counter/AtomicCounter.java
+++ b/src/{project}/counter/AtomicCounter.java
@@ -18,7 +18,5 @@ public class AtomicCounter {{
-    private AtomicInteger count = new AtomicInteger(0);
+    private int count = 0;

     public int increment() {{
-        return count.incrementAndGet();
+        return ++count;
     }}
 }}""",
        ],
        'log_templates': [
            ["WARN [CacheManager] ConcurrentModificationException",
             "HashMap modified during iteration",
             "Race condition detected in cache operations"],
            ["ERROR [AtomicCounter] Counter value inconsistent",
             "Expected: 10000, Actual: 9847",
             "Lost updates due to race condition"],
        ]
    }
}

# Normal commit templates (non-bug changes)
NORMAL_PATTERNS = {
    'refactor': [
        """--- a/src/{project}/util/StringUtils.java
+++ b/src/{project}/util/StringUtils.java
@@ -10,6 +10,10 @@ public class StringUtils {{
+    /**
+     * Checks if string is empty or null.
+     * @param str the string to check
+     * @return true if empty or null
+     */
     public static boolean isEmpty(String str) {{
         return str == null || str.length() == 0;
     }}
 }}""",
        """--- a/src/{project}/model/User.java
+++ b/src/{project}/model/User.java
@@ -5,6 +5,8 @@ public class User {{
     private String name;
+    private String email;
+    private LocalDateTime createdAt;

     public String getName() {{
         return name;
     }}
+
+    public String getEmail() {{
+        return email;
+    }}
 }}""",
    ],
    'feature': [
        """--- a/src/{project}/api/UserController.java
+++ b/src/{project}/api/UserController.java
@@ -45,6 +45,15 @@ public class UserController {{
+    @GetMapping("/users/search")
+    public ResponseEntity<List<User>> searchUsers(
+            @RequestParam String query,
+            @RequestParam(defaultValue = "10") int limit) {{
+        List<User> users = userService.search(query, limit);
+        return ResponseEntity.ok(users);
+    }}
+
     @GetMapping("/users/{{id}}")
     public ResponseEntity<User> getUser(@PathVariable Long id) {{
 }}""",
        """--- a/src/{project}/service/NotificationService.java
+++ b/src/{project}/service/NotificationService.java
@@ -22,4 +22,18 @@ public class NotificationService {{
+    public void sendBatchNotifications(List<User> users, String message) {{
+        users.parallelStream().forEach(user -> {{
+            try {{
+                sendEmail(user.getEmail(), message);
+                log.info("Notification sent to {{}}", user.getEmail());
+            }} catch (Exception e) {{
+                log.error("Failed to send to {{}}", user.getEmail(), e);
+            }}
+        }});
+    }}
 }}""",
    ],
    'test': [
        """--- a/test/{project}/util/StringUtilsTest.java
+++ b/test/{project}/util/StringUtilsTest.java
@@ -15,6 +15,16 @@ public class StringUtilsTest {{
+    @Test
+    public void testIsEmpty_withNull_returnsTrue() {{
+        assertTrue(StringUtils.isEmpty(null));
+    }}
+
+    @Test
+    public void testIsEmpty_withEmptyString_returnsTrue() {{
+        assertTrue(StringUtils.isEmpty(""));
+    }}
+
     @Test
     public void testIsEmpty_withNonEmpty_returnsFalse() {{
 }}""",
    ],
    'config': [
        """--- a/src/{project}/config/AppConfig.java
+++ b/src/{project}/config/AppConfig.java
@@ -8,6 +8,7 @@ public class AppConfig {{
     private int port = 8080;
+    private int maxConnections = 100;
+    private Duration timeout = Duration.ofSeconds(30);

     @Bean
     public DataSource dataSource() {{
 }}""",
    ],
}

# Projects for diversity
PROJECTS = [
    'openstack', 'hadoop', 'kubernetes', 'tensorflow', 'spark',
    'elasticsearch', 'cassandra', 'kafka', 'flink', 'hbase',
    'spring-boot', 'netty', 'dubbo', 'rocketmq', 'nacos',
    'guava', 'jackson', 'gson', 'fastjson', 'lombok'
]


def generate_bug_sample(project: str, bug_id: int, bug_type: str, add_noise: bool = True) -> Dict:
    """Generate a bug-inducing code change sample."""
    pattern = BUG_PATTERNS[bug_type]

    diff_template = random.choice(pattern['diff_templates'])
    log_template = random.choice(pattern['log_templates'])

    diff = diff_template.format(project=project.capitalize())

    # Add some variation to logs
    logs = []
    timestamp_base = 1700000000 + random.randint(0, 10000000)

    # Mix in normal logs to make it harder (70% chance to add normal logs first)
    if add_noise and random.random() < 0.7:
        normal_logs = [
            "INFO [Application] Processing request",
            "DEBUG [Service] Cache lookup completed",
            "INFO [Database] Query executed in 23ms",
            "INFO [Scheduler] Background task running",
        ]
        for i, msg in enumerate(random.sample(normal_logs, random.randint(1, 3))):
            logs.append({
                "timestamp": timestamp_base + i * random.randint(60, 300),
                "level": msg.split()[0],
                "message": msg
            })

    # Add the error logs (but only 60% of bug samples have clear error logs)
    if random.random() < 0.6:
        for i, msg in enumerate(log_template):
            logs.append({
                "timestamp": timestamp_base + len(logs) * random.randint(1, 60) + 1000,
                "level": msg.split()[0] if msg.split()[0] in ['ERROR', 'WARN', 'FATAL', 'INFO'] else 'ERROR',
                "message": msg
            })
    else:
        # Some bugs don't have clear error logs (latent bugs)
        subtle_logs = [
            "WARN [Monitor] Response time increased to 250ms",
            "INFO [Service] Retry attempt 1 of 3",
            "DEBUG [Handler] Unexpected null value handled",
            "INFO [Application] Request completed with warnings",
        ]
        for i, msg in enumerate(random.sample(subtle_logs, random.randint(1, 2))):
            logs.append({
                "timestamp": timestamp_base + len(logs) * random.randint(60, 300) + 1000,
                "level": msg.split()[0],
                "message": msg
            })

    return {
        "id": f"bug_{project}_{bug_id}_{bug_type}",
        "code_change": {
            "project": project,
            "commit_sha": hashlib.sha1(f"{project}_{bug_id}".encode()).hexdigest()[:8],
            "diff": diff,
            "bug_type": bug_type,
            "additions": diff.count('\n+') - diff.count('+++'),
            "deletions": diff.count('\n-') - diff.count('---')
        },
        "log_context": {
            "messages": logs,
            "window_hours": random.choice([1, 6, 12, 24]),
            "total_entries": len(logs) + random.randint(50, 500)
        },
        "label": 1,
        "source": "synthetic_bug"
    }


def generate_normal_sample(project: str, commit_id: int, change_type: str, add_noise: bool = True) -> Dict:
    """Generate a normal (non-bug) code change sample."""
    templates = NORMAL_PATTERNS[change_type]
    diff_template = random.choice(templates)
    diff = diff_template.format(project=project.capitalize())

    # Normal commits have routine logs
    logs = []
    timestamp_base = 1700000000 + random.randint(0, 10000000)
    routine_logs = [
        "INFO [Application] Server started on port 8080",
        "INFO [Database] Connection pool initialized with 10 connections",
        "DEBUG [UserService] Processing request for user lookup",
        "INFO [Cache] Cache hit ratio: 0.85",
        "INFO [Scheduler] Scheduled task completed successfully",
        "DEBUG [RequestHandler] Request processed in 45ms"
    ]

    for i, msg in enumerate(random.sample(routine_logs, min(3, len(routine_logs)))):
        logs.append({
            "timestamp": timestamp_base + i * random.randint(60, 300),
            "level": msg.split()[0],
            "message": msg
        })

    # Add some warning/error logs to normal samples (30% chance) - these are transient issues
    if add_noise and random.random() < 0.3:
        transient_issues = [
            "WARN [Connection] Connection timeout, retrying...",
            "WARN [Cache] Cache miss for key user_123",
            "ERROR [Network] Temporary network error, recovered",
            "WARN [GC] Long GC pause detected: 150ms",
            "WARN [ThreadPool] Thread pool near capacity: 95%",
            "ERROR [External] External service returned 503, using fallback",
        ]
        for i, msg in enumerate(random.sample(transient_issues, random.randint(1, 2))):
            logs.append({
                "timestamp": timestamp_base + len(logs) * random.randint(60, 300) + 500,
                "level": msg.split()[0],
                "message": msg
            })

    return {
        "id": f"normal_{project}_{commit_id}_{change_type}",
        "code_change": {
            "project": project,
            "commit_sha": hashlib.sha1(f"normal_{project}_{commit_id}".encode()).hexdigest()[:8],
            "diff": diff,
            "change_type": change_type,
            "additions": diff.count('\n+') - diff.count('+++'),
            "deletions": max(0, diff.count('\n-') - diff.count('---'))
        },
        "log_context": {
            "messages": logs,
            "window_hours": random.choice([1, 6, 12, 24]),
            "total_entries": len(logs) + random.randint(100, 1000)
        },
        "label": 0,
        "source": "synthetic_normal"
    }


def generate_hard_bug_sample(project: str, bug_id: int, bug_type: str) -> Dict:
    """Generate a harder bug sample with subtle patterns."""
    pattern = BUG_PATTERNS[bug_type]
    diff_template = random.choice(pattern['diff_templates'])
    diff = diff_template.format(project=project.capitalize())

    # Randomize the diff more - add noise tokens
    noise_tokens = ['// TODO', '// FIXME', '/* comment */', 'import java.util.*;',
                    '@Override', '@Deprecated', 'public', 'private', 'static']
    if random.random() < 0.5:
        noise = random.choice(noise_tokens)
        diff = diff.replace('public class', f'{noise}\npublic class')

    # Most bug samples have NO clear error logs (realistic scenario)
    logs = []
    timestamp_base = 1700000000 + random.randint(0, 10000000)

    # Add normal-looking logs
    normal_logs = [
        "INFO [Application] Processing request",
        "DEBUG [Service] Cache lookup completed",
        "INFO [Database] Query executed in 23ms",
        "INFO [Scheduler] Background task running",
        "DEBUG [Handler] Request validated",
        "INFO [Service] Operation completed",
    ]
    for i, msg in enumerate(random.sample(normal_logs, random.randint(2, 4))):
        logs.append({
            "timestamp": timestamp_base + i * random.randint(60, 300),
            "level": msg.split()[0],
            "message": msg
        })

    # Only 20% have clear error logs (most bugs are latent)
    if random.random() < 0.2:
        log_template = random.choice(pattern['log_templates'])
        for msg in log_template[:1]:  # Only add 1 error log
            logs.append({
                "timestamp": timestamp_base + len(logs) * 60 + 1000,
                "level": "WARN",  # Downgrade to WARN
                "message": msg.replace("ERROR", "WARN").replace("FATAL", "WARN")
            })

    return {
        "id": f"bug_{project}_{bug_id}_{bug_type}",
        "code_change": {
            "project": project,
            "commit_sha": hashlib.sha1(f"{project}_{bug_id}".encode()).hexdigest()[:8],
            "diff": diff,
            "bug_type": bug_type,
            "additions": diff.count('\n+') - diff.count('+++'),
            "deletions": diff.count('\n-') - diff.count('---')
        },
        "log_context": {
            "messages": logs,
            "window_hours": random.choice([1, 6, 12, 24]),
            "total_entries": len(logs) + random.randint(50, 500)
        },
        "label": 1,
        "source": "synthetic_bug_hard"
    }


def generate_hard_normal_sample(project: str, commit_id: int, change_type: str) -> Dict:
    """Generate a harder normal sample that looks similar to bugs."""
    templates = NORMAL_PATTERNS[change_type]
    diff_template = random.choice(templates)
    diff = diff_template.format(project=project.capitalize())

    # Sometimes use bug-like diff patterns for normal commits (false positive challenge)
    if random.random() < 0.3:
        bug_type = random.choice(list(BUG_PATTERNS.keys()))
        bug_diff = random.choice(BUG_PATTERNS[bug_type]['diff_templates'])
        # But reverse the change (fix instead of introduce)
        diff = bug_diff.format(project=project.capitalize())
        diff = diff.replace('---', 'XXX').replace('+++', '---').replace('XXX', '+++')
        diff = diff.replace('\n-', '\nYYY').replace('\n+', '\n-').replace('\nYYY', '\n+')

    logs = []
    timestamp_base = 1700000000 + random.randint(0, 10000000)

    # Normal commits can also have warnings/errors (transient issues)
    all_logs = [
        "INFO [Application] Server started on port 8080",
        "INFO [Database] Connection pool initialized",
        "DEBUG [UserService] Processing request",
        "INFO [Cache] Cache hit ratio: 0.85",
        "WARN [Connection] Connection timeout, retrying...",
        "WARN [GC] Long GC pause detected: 150ms",
        "ERROR [Network] Temporary network error, recovered",
        "INFO [Service] Retry successful",
    ]

    for i, msg in enumerate(random.sample(all_logs, random.randint(3, 5))):
        logs.append({
            "timestamp": timestamp_base + i * random.randint(60, 300),
            "level": msg.split()[0],
            "message": msg
        })

    return {
        "id": f"normal_{project}_{commit_id}_{change_type}",
        "code_change": {
            "project": project,
            "commit_sha": hashlib.sha1(f"normal_{project}_{commit_id}".encode()).hexdigest()[:8],
            "diff": diff,
            "change_type": change_type,
            "additions": diff.count('\n+') - diff.count('+++'),
            "deletions": max(0, diff.count('\n-') - diff.count('---'))
        },
        "log_context": {
            "messages": logs,
            "window_hours": random.choice([1, 6, 12, 24]),
            "total_entries": len(logs) + random.randint(100, 1000)
        },
        "label": 0,
        "source": "synthetic_normal_hard"
    }


def generate_dataset(
    n_bugs: int = 500,
    n_normal: int = 500,
    seed: int = 42,
    difficulty: str = 'hard'
) -> List[Dict]:
    """Generate complete dataset.

    Args:
        difficulty: 'easy' for original, 'hard' for realistic difficulty
    """
    random.seed(seed)

    samples = []
    bug_types = list(BUG_PATTERNS.keys())
    normal_types = list(NORMAL_PATTERNS.keys())

    # Generate bug samples
    for i in range(n_bugs):
        project = random.choice(PROJECTS)
        bug_type = random.choice(bug_types)
        if difficulty == 'hard':
            sample = generate_hard_bug_sample(project, i, bug_type)
        else:
            sample = generate_bug_sample(project, i, bug_type)
        samples.append(sample)

    # Generate normal samples
    for i in range(n_normal):
        project = random.choice(PROJECTS)
        change_type = random.choice(normal_types)
        if difficulty == 'hard':
            sample = generate_hard_normal_sample(project, i, change_type)
        else:
            sample = generate_normal_sample(project, i, change_type)
        samples.append(sample)

    random.shuffle(samples)
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test."""
    random.seed(seed)

    # Stratified split by label
    positives = [s for s in samples if s['label'] == 1]
    negatives = [s for s in samples if s['label'] == 0]

    random.shuffle(positives)
    random.shuffle(negatives)

    def split_list(lst, train_r, val_r):
        n = len(lst)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    pos_train, pos_val, pos_test = split_list(positives, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(negatives, train_ratio, val_ratio)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic data for CodeFault")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent.parent / "data" / "processed" / "codefault")
    parser.add_argument("--n-bugs", type=int, default=800,
                       help="Number of bug samples")
    parser.add_argument("--n-normal", type=int, default=400,
                       help="Number of normal samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default='hard',
                       choices=['easy', 'hard'],
                       help="Difficulty level: 'easy' for obvious patterns, 'hard' for realistic")

    args = parser.parse_args()

    print("=" * 60)
    print("CodeFault Synthetic Data Generator")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Difficulty: {args.difficulty}")
    print("=" * 60)

    # Generate data
    print(f"\nGenerating {args.n_bugs} bug samples and {args.n_normal} normal samples...")
    samples = generate_dataset(args.n_bugs, args.n_normal, args.seed, difficulty=args.difficulty)

    # Split
    print("\nSplitting into train/val/test...")
    train, val, test = split_dataset(samples, seed=args.seed)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [('train', train), ('val', val), ('test', test)]:
        filepath = args.output_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        pos = sum(1 for s in data if s['label'] == 1)
        neg = len(data) - pos
        print(f"  {name}: {len(data)} samples (pos={pos}, neg={neg})")

    # Metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "n_bugs": args.n_bugs,
        "n_normal": args.n_normal,
        "total_samples": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "positive_ratio": sum(1 for s in samples if s['label'] == 1) / len(samples),
        "bug_types": list(BUG_PATTERNS.keys()),
        "projects": PROJECTS
    }

    with open(args.output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to {args.output_dir}")
    print(f"Total: {len(samples)} samples, Positive ratio: {metadata['positive_ratio']:.1%}")


if __name__ == "__main__":
    main()
