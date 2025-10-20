import React, { ReactNode } from 'react';
import clsx from 'clsx';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
  gradient?: boolean;
}

export function Card({ title, children, className, action, gradient = false }: CardProps) {
  return (
    <div 
      className={clsx(
        'bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden transition-all duration-200 hover:shadow-md',
        className
      )}
    >
      {title && (
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-lg font-bold text-slate-900">
            {title}
          </h3>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
}

