from utilities.config import create_folders,generator,discriminator,OptimizerD,OptimizerG,l1_lambda,BCE_Loss,L1_Loss
from utilities.config import num_of_epochs,device
from data.dataset import train_loader,val_loader
from tqdm import tqdm
import torch
from torchvision.utils import save_image


create_folders()

class Trainer:
    def __init__(self,generator,discriminator,disc_optimizer,gen_optimizer,generator_loss,discriminator_loss):
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator = generator
        self.discriminator=discriminator
        print("Training Started")

    def get_disc_loss(self,output,real=True):
        if real:
            disc_loss = self.discriminator_loss(output,torch.ones_like(output))
        else:
            disc_loss = self.discriminator_loss(output,torch.zeros_like(output))

        return disc_loss


    def train(self):
        print("Training Started")
        for epoch in range(num_of_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                for idx, (x,y) in enumerate(tepoch):
                    x,y = x.to(device),y.to(device)
                    y_fake = self.generator(x)
                    discriminator_real =self.discriminator(x,y)
                    discriminator_fake = self.discriminator(x,y_fake.detach())
                    discriminator_real_loss =  self.get_disc_loss(discriminator_real,True)
                    discriminator_fake_loss = self.get_disc_loss(discriminator_fake)
                    discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) /2

                    self.discriminator.zero_grad()
                    discriminator_loss.backward()
                    self.disc_optimizer.step()

                    D_fake = self.discriminator(x,y_fake)
                    G_fake_loss = self.get_disc_loss(D_fake)
                    l1_loss = self.generator_loss(y_fake,y) * l1_lambda
                    generator_loss = G_fake_loss * l1_loss

                    self.generator.zero_grad()
                    generator_loss.backward()
                    self.gen_optimizer.step()
                    tepoch.set_postfix(**{"Discriminator Loss" : discriminator_loss.item(), "Generator Loss ":generator_loss.item()})
                if epoch % 50==0:
                    self.save_some_examples(self.generator, val_loader, epoch, "generated","input","label")
                    self.save_checkpoint(self.generator,self.gen_optimizer,f"generator_checkpoint_{epoch}.pt")
                    self.save_checkpoint(self.discriminator,self.disc_optimizer,f"discriminator_checkpoint_{epoch}.pt")

    def save_checkpoint(self,model, optimizer, filename):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def save_some_examples(self,gen, val_loader, epoch, folder_for_generated,folder_for_input,folder_for_label):
        x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake *255.0
            save_image(y_fake, folder_for_generated + f"/y_gen_{epoch}.png")
            save_image(x*0.5 + 0.5, folder_for_input + f"/input_{epoch}.png")
            save_image(y*0.5 + 0.5, folder_for_label + f"/label_{epoch}.png")
        gen.train()


if __name__ =="__main__":
    trainer = Trainer(generator,discriminator,OptimizerD,OptimizerG,L1_Loss,BCE_Loss)
    trainer.train()