import React from 'react';

export const NavBar = () => {
  return (
    <nav className="bg-indigo-600 shadow-md">
      <div className="container mx-auto px-4 py-2 flex justify-between items-center">
        {/* Logo */}
        <div className="text-white text-xl font-bold">
          <span role="img" aria-label="robot">🤖</span> iaChat
        </div>
      </div>
    </nav>
  );
};
