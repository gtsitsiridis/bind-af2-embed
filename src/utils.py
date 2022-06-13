class FileManager(object):
    @staticmethod
    def read_ids(ids_file) -> list:
        ids = []
        with open(ids_file) as read_in:
            for line in read_in:
                ids.append(line.strip())
        return ids

    @staticmethod
    def read_split_ids(splits_ids_files: list) -> (list, list):
        """

        :param splits_ids_files:
        :return:
        """
        ids = []
        fold_array = []
        s = 1
        for ids_file in splits_ids_files:
            split_ids = FileManager.read_ids(ids_file)
            ids += split_ids
            fold_array += [s] * len(split_ids)
            s += 1
        return ids, fold_array

    @staticmethod
    def read_binding_residues(file_in):
        """
        Read binding residues from file
        :param file_in:
        :return:
        """
        binding = dict()

        with open(file_in) as read_in:
            for line in read_in:
                splitted_line = line.strip().split()
                if len(splitted_line) > 1:
                    identifier = splitted_line[0]
                    residues = splitted_line[1].split(',')
                    residues_int = [int(r) for r in residues]

                    binding[identifier] = residues_int

        return binding