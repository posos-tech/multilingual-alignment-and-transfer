import random


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", type=str, nargs="+")
    parser.add_argument("--output_files", type=str, nargs="+")
    parser.add_argument("--num_samples", type=int, default=1_000_000)
    args = parser.parse_args()

    line_positions = []

    input_files = args.input_files
    output_files = args.output_files

    if not isinstance(input_files, list):
        input_files = [input_files]
    if not isinstance(output_files, list):
        output_files = [output_files]

    assert len(input_files) == len(output_files)

    class FileList:
        def __init__(self, fnames: list, mode="r"):
            self.fnames = fnames
            self.mode = mode

        def __enter__(self):
            self.files = list(map(lambda x: open(x, self.mode), self.fnames))
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            for f in self.files:
                f.close()

        def __iter__(self):
            for lines in zip(*self.files):
                yield list(lines)

        def tell(self):
            return [f.tell() for f in self.files]

        def seek(self, positions: list):
            for f, pos in zip(self.files, positions):
                f.seek(pos)

        def readlines(self):
            return [f.readline() for f in self.files]

        def write(self, lines: list):
            for f, line in zip(self.files, lines):
                f.write(line)

    with FileList(input_files, "rb") as files_reader:
        positions = files_reader.tell()
        for i, lines in enumerate(files_reader):
            if i < args.num_samples:
                line_positions.append(positions)
            else:
                j = random.randrange(i + 1)
                if j < args.num_samples:
                    line_positions[j] = positions
            positions = files_reader.tell()

        with FileList(output_files, "wb") as files_writer:
            for pos_list in line_positions:
                files_reader.seek(pos_list)
                files_writer.write(files_reader.readlines())
