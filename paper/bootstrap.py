import tqdm
from multiprocessing import Process, Pipe


def bootstrap(
    func, args, kwargs,
    num_runs,
    num_jobs = 1,
):

    if num_jobs == 1:
        for iter_id in tqdm.tqdm(range(num_runs)):
            kwargs['seed'] = iter_id
            func(*args, **kwargs)

    elif num_jobs > 1:
        # initialize pipes
        pipe_list = [Pipe() for i in range(num_jobs)]

        # initialize processes
        proc_list = [
            Process(
                target = _mp_wrapper, 
                args = (pipe_list[i][1], func, args, kwargs)
            ) for i in range(num_jobs)
        ]

        # start subprocesses
        for i in range(num_jobs):
            proc_list[i].start()

        # main loop
        for iter_id in tqdm.tqdm(range(num_runs)):
            # rotate
            i = 0
            while True:
                if pipe_list[i][0].poll():
                    # receive ready signal
                    _ = pipe_list[i][0].recv()

                    # send iter_id to subprocess
                    seed = iter_id
                    pipe_list[i][0].send(seed)

                    # next iter_id
                    break

                else:
                    i = (i + 1) % num_jobs

        # send termination signal to subprocesses
        for i in range(num_jobs):
            pipe_list[i][0].send(-1)

        # close pipes
        for i in range(num_jobs):
            pipe_list[i][0].close()

        # waiting for subprocesses to join/return
        for i in range(num_jobs):
            proc_list[i].join()

    else:
        pass


def _mp_wrapper(pipe, func, args, kwargs):

    while True:
        # send ready signal to main process
        pipe.send(1)

        # receive iter_id from main process
        seed = pipe.recv()

        if seed != -1:
            # seed can be any number as long as it's unique among all subprocesses
            kwargs['seed'] = seed

            # run
            func(*args, **kwargs)

        else:
            pipe.close()
            return
