// SHURA (HEAPBUG1): the SIGABRT handler prints its marker and the backtraces, then the process still dies by SIGABRT.
// The child corrupts a heap chunk's size field and frees it: glibc aborts exactly like the "corrupted size vs. prev_size" crash of the stand.
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static int child() {
    common_install_abort_backtrace();
    if (getenv("TEST_ABORT_BACKTRACE_GDB") == nullptr) {
        setenv("GGML_NO_BACKTRACE", "1", 1);   // the test checks the handler, not gdb; the real server leaves this unset
    }
    char * p = (char *) malloc(48);
    volatile size_t * hdr = (volatile size_t *) (p - sizeof(size_t));   // the chunk's own size field (volatile: the write must survive)
    *hdr = 0x7ffffffffffffff0ull;                                        // glibc's free() notices the impossible size and aborts
    free(p);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "child") == 0) {
        return child();
    }
    int fds[2];
    if (pipe(fds) != 0) { return 2; }
    const pid_t pid = fork();
    if (pid < 0) { return 2; }
    if (pid == 0) {
        dup2(fds[1], 2);
        close(fds[0]);
        execl(argv[0], argv[0], "child", (char *) nullptr);
        _exit(3);
    }
    close(fds[1]);
    std::string err;
    char buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) { err.append(buf, n); }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    const bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    const bool marker = err.find("fatal: SIGABRT") != std::string::npos && err.find("re-raising SIGABRT") != std::string::npos;
    const bool glibc = err.find("free()") != std::string::npos || err.find("corrupted") != std::string::npos || err.find("malloc") != std::string::npos;
    printf("exit by SIGABRT: %d, handler marker: %d, glibc message: %d\n", aborted, marker, glibc);
    if (!aborted || !marker) {
        fprintf(stderr, "--- child stderr ---\n%s\n", err.c_str());
        return 1;
    }
    return 0;
}
