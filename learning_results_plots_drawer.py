import matplotlib.pyplot as plt
import os


class Epoch:
    def __init__(self, epoch_nr, train_loss, train_acc, val_loss, val_acc):
        self.epoch_nr = epoch_nr
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc


class PlotDrawer:
    def __init__(self):
        self.epochs = []

    def parse(self, file_name):
        self.x_label = "epoch nr"
        file_name_after_strip = file_name.strip("output.h5")
        file_name_after_split = file_name_after_strip.split('/')
        self.title = file_name_after_split[len(file_name_after_split) - 1] + "plot"
        print(self.title)

        file = open(file_name, "r")
        for line in file:
            line = line.strip('\n')
            if line[0:5] == "Epoch":
                line_after_split = line.split(' ')
                line_after_split_2 = line_after_split[1].split('/')
                epoch_nr = line_after_split_2[0]
            if line[0:5] == "train":
                line_after_split = line.split(' ')
                train_loss = line_after_split[2]
                train_acc = line_after_split[4]
            if line[0:3] == "val":
                line_after_split = line.split(' ')
                val_loss = line_after_split[2]
                val_acc = line_after_split[4]
                epoch = Epoch(epoch_nr, train_loss, train_acc, val_loss, val_acc)
                self.epochs.append(epoch)
                # print("nr: "+epoch.epoch_nr+", train_loss:"+
                #   epoch.train_loss+", train_acc: "+epoch.train_acc+
                #   ",val_loss: "+epoch.val_loss+", val_acc: "+epoch.val_acc+"\n")

    def draw_plot(self):
        x_epoch = []
        y_train_loss = []
        y_train_acc = []
        y_val_loss = []
        y_val_acc = []
        for epoch in self.epochs:
            x_epoch.append(epoch.epoch_nr)
            y_train_loss.append(float(epoch.train_loss))
            y_train_acc.append(float(epoch.train_acc))
            y_val_loss.append(float(epoch.val_loss))
            y_val_acc.append(float(epoch.val_acc))

        plt.clf()
        plt.title(self.title)
        plt.grid(True)
        plt.xlabel(self.x_label)
        plt.ylim(0.1, 1.0)

        plt.plot(x_epoch, y_train_loss)
        plt.plot(x_epoch, y_train_acc)
        plt.plot(x_epoch, y_val_loss)
        plt.plot(x_epoch, y_val_acc)
        plt.legend(["train loss", "train accuracy", "val loss", "val accuracy"])
        #plt.savefig("/content/drive/My Drive/monika/" + self.title, dpi=72)
        plt.savefig("plots/"+self.title, dpi=72)


def main():
    plot_drawer = PlotDrawer()
    #plot_drawer.parse("/content/drive/My Drive/monika/2a_output.h5")
    plot_drawer.parse("outputs/2a_output.h5")
    plot_drawer.draw_plot()


if __name__ == "__main__":
    main()