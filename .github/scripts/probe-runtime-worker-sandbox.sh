#!/usr/bin/env bash
set -euo pipefail

if ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "probe-runtime-worker-sandbox: sandbox-exec not found" >&2
  exit 1
fi

if ! command -v clang >/dev/null 2>&1; then
  echo "probe-runtime-worker-sandbox: clang not found" >&2
  exit 1
fi

root="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/mlxfast-sandbox-probe.XXXXXX")"
root="$(cd -P "${root}" && pwd)"

probe_source="${root}/probe.c"
probe_bin="${root}/probe"
profile_path="${root}/worker.sb"
private_dir="${root}/private"
golden_path="${private_dir}/correctness_golden.json"
private_path="${private_dir}/gpqa_reference_cases.json"
outside_write_path="${root}/outside-write.txt"
private_write_path="${private_dir}/private-write.txt"
unix_socket_path="${root}/probe.sock"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
  rm -rf "${root}"
}

mkdir -p "${private_dir}"
printf '%s\n' '{"secret":true}' > "${golden_path}"
printf '%s\n' '{"secret":true}' > "${private_path}"
trap cleanup EXIT

cat > "${probe_source}" <<'C'
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static int is_denied_errno(int value) {
  return value == EACCES || value == EPERM;
}

static void fail_errno(const char *label) {
  fprintf(stderr, "%s: unexpected errno=%d (%s)\n", label, errno, strerror(errno));
  exit(1);
}

static void expect_read_denied(const char *label, const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd >= 0) {
    close(fd);
    fprintf(stderr, "%s: read unexpectedly succeeded\n", label);
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno(label);
  }
}

static void expect_write_denied(const char *label, const char *path) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd >= 0) {
    close(fd);
    fprintf(stderr, "%s: write unexpectedly succeeded\n", label);
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno(label);
  }
}

static void expect_dev_null_write_allowed(void) {
  int fd = open("/dev/null", O_WRONLY);
  if (fd < 0) {
    fail_errno("/dev/null write");
  }
  if (write(fd, "x", 1) != 1) {
    fail_errno("/dev/null write payload");
  }
  close(fd);
}

static void expect_inet_network_denied(void) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("inet socket");
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(9);
  int result = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
  int saved_errno = errno;
  close(fd);
  if (result < 0 && is_denied_errno(saved_errno)) {
    return;
  }

  fprintf(stderr, "inet network unexpectedly reached socket/connect path errno=%d (%s)\n", saved_errno, strerror(saved_errno));
  exit(1);
}

static void expect_unix_network_denied(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("unix socket");
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t length = strlen(path);
  if (length >= sizeof(addr.sun_path)) {
    fprintf(stderr, "unix socket path too long\n");
    exit(1);
  }
  memcpy(addr.sun_path, path, length + 1);
  int result = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
  int saved_errno = errno;
  close(fd);
  if (result < 0 && is_denied_errno(saved_errno)) {
    return;
  }

  fprintf(stderr, "unix socket connect unexpectedly reached path errno=%d (%s)\n", saved_errno, strerror(saved_errno));
  exit(1);
}

static void expect_fork_denied(void) {
  pid_t pid = fork();
  if (pid < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("fork");
  }
  if (pid == 0) {
    _exit(33);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  fprintf(stderr, "fork unexpectedly succeeded\n");
  exit(1);
}

static void expect_spawn_denied(void) {
  pid_t pid = 0;
  char *argv[] = {"/bin/echo", "sandbox-probe", NULL};
  int result = posix_spawn(&pid, "/bin/echo", NULL, NULL, argv, environ);
  if (result != 0) {
    if (is_denied_errno(result)) {
      return;
    }
    fprintf(stderr, "posix_spawn: unexpected errno=%d (%s)\n", result, strerror(result));
    exit(1);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  fprintf(stderr, "posix_spawn unexpectedly succeeded\n");
  exit(1);
}

static int run_server(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    fail_errno("server socket");
  }
  (void)unlink(path);
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t length = strlen(path);
  if (length >= sizeof(addr.sun_path)) {
    fprintf(stderr, "server unix socket path too long\n");
    return 2;
  }
  memcpy(addr.sun_path, path, length + 1);
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    fail_errno("server bind");
  }
  if (listen(fd, 1) < 0) {
    fail_errno("server listen");
  }
  for (;;) {
    pause();
  }
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--server") == 0) {
    return run_server(argv[2]);
  }
  if (argc != 6) {
    fprintf(stderr, "usage: %s golden private outside-write private-write unix-socket\n", argv[0]);
    return 2;
  }
  expect_read_denied("golden file", argv[1]);
  expect_read_denied("private file", argv[2]);
  expect_write_denied("outside write", argv[3]);
  expect_write_denied("private write", argv[4]);
  expect_dev_null_write_allowed();
  expect_inet_network_denied();
  expect_unix_network_denied(argv[5]);
  expect_fork_denied();
  expect_spawn_denied();
  printf("runtime worker sandbox probe passed\n");
  return 0;
}
C

clang "${probe_source}" -o "${probe_bin}"

"${probe_bin}" --server "${unix_socket_path}" &
server_pid="$!"
for _ in {1..50}; do
  if [[ -S "${unix_socket_path}" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -S "${unix_socket_path}" ]]; then
  echo "probe-runtime-worker-sandbox: unix socket listener did not start" >&2
  exit 1
fi

cat > "${profile_path}" <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "${probe_bin}"))
(deny file-write*)
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "${golden_path}"))
(deny file-read* (subpath "${private_dir}"))
(deny file-write* (subpath "${private_dir}"))
EOF

sandbox-exec -f "${profile_path}" "${probe_bin}" \
  "${golden_path}" \
  "${private_path}" \
  "${outside_write_path}" \
  "${private_write_path}" \
  "${unix_socket_path}"
